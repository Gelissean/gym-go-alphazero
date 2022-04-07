from adaptive_rl.net import Net, ResidualBlock, weights_init_xavier
from adaptive_rl.replay_buffer import ExperienceReplay
import torch
from gym_go import govars

class Experience_buffer_GO(ExperienceReplay):
    def __init__(self, size, height, width, batch_size):
        self.states = torch.zeros((size, govars.NUM_CHNLS, height, width), dtype = torch.int16)
        self.policies = torch.zeros((size, height * width+1))
        self.values = torch.zeros((size, 1), dtype = torch.long)
        self.moves_left = torch.zeros((size, 1), dtype = torch.long)

        self.max_size = size
        self.index = 0
        self.batch_size = batch_size

        self.full_buffer = False

        self.height = height
        self.width = width
        self.actions = height * width+1

    def add(self, length, states, policies, values, moves_left):
        states, policies = states.cpu(), policies.cpu()

        new_index = self.index + length
        if new_index <= self.max_size:
            self.states[self.index:new_index] = states
            self.policies[self.index:new_index] = policies
            self.values[self.index:new_index] = values
            self.moves_left[self.index:new_index] = moves_left
            self.index = new_index
            if self.index == self.max_size:
                self.index = 0
                self.full_buffer = True
        else:
            to_end = self.max_size - self.index
            from_start = length - to_end
            self.states[self.index:self.max_size] = states[0:to_end]
            self.policies[self.index:self.max_size] = policies[0:to_end]
            self.values[self.index:self.max_size] = values[0:to_end]
            self.moves_left[self.index:self.max_size] = moves_left[0:to_end]
            self.states[0:from_start] = states[to_end:length]
            self.policies[0:from_start] = policies[to_end:length]
            self.values[0:from_start] = values[to_end:length]
            self.moves_left[0:from_start] = moves_left[to_end:length]
            self.index = from_start
            self.full_buffer = True

    def store(self, states, policies, v): #Bx1x5x5, #Bx1x25, #Bx1x25
        length, dim1, dim2, dim3 = states.shape
        values = torch.ones((length, 1), dtype = torch.long)
        moves = torch.arange(length).long().flip(0).view(-1, 1)
        if v > 0.0:
            indices_even = torch.arange(0, length, 2)
            indices_odd = torch.arange(1, length, 2)

            if length % 2 == 0:
                values[indices_even, 0] = 0
                values[indices_odd, 0] = 2
            else:
                values[indices_even, 0] = 2
                values[indices_odd, 0] = 0
        elif v < 0.0:
            indices_even = torch.arange(0, length, 2)
            indices_odd = torch.arange(1, length, 2)

            if length % 2 == 0:
                values[indices_even, 0] = 2
                values[indices_odd, 0] = 0
            else:
                values[indices_even, 0] = 0
                values[indices_odd, 0] = 2

        self.add(length, states, policies, values, moves)

        policies_pass = policies[:, -1].view(-1, 1)
        policies = policies[:, :-1].view(-1, 1, self.height, self.width)
        self.add(length, torch.flip(states, [3]), torch.cat((torch.flip(policies, [3]).view(-1, self.actions-1), policies_pass), 1), values, moves)

        for i in range(3):
            states, policies = torch.rot90(states, 1, [2, 3]), torch.rot90(policies, 1, [2, 3])
            self.add(length, states, torch.cat((policies.reshape(-1, self.actions - 1), policies_pass), 1), values, moves)
            self.add(length, torch.flip(states, [3]), torch.cat((torch.flip(policies, [3]).reshape(-1, self.actions-1), policies_pass),1), values, moves)

    def sample(self):
        if self.full_buffer:
            indices = np.random.choice(self.max_size, self.batch_size)
        else:
            indices = np.random.choice(self.index, self.batch_size)
        return self.states[indices], self.policies[indices], self.values[indices], self.moves_left[indices]


import torch.nn.init as init

def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        init.orthogonal_(m.weight)

class Net_GO(Net):
    def __init__(self, input_frames, blocks, filters, size, actions):
        super(Net, self).__init__()
        self.blocks = blocks
        self.size = size
        self.features_count = size * 32
        self.actions = actions

        self.conv1 = nn.Conv2d(input_frames, filters, 3, stride=1, padding=1)
        self.residual_blocks = nn.ModuleList([ResidualBlock(filters) for i in range(blocks)])

        self.conv_policy = nn.Conv2d(filters, 32, 3, stride=1, padding=1)
        self.fc1_p = nn.Linear(self.features_count, 256)
        self.fc2_p = nn.Linear(256, self.actions)

        self.conv_value = nn.Conv2d(filters, 32, 3, stride=1, padding=1)
        self.fc1_v = nn.Linear(self.features_count, 128)
        self.fc2_v = nn.Linear(128, 3)

        self.conv_moves_left = nn.Conv2d(filters, 32, 3, stride=1, padding=1)
        self.fc1_m = nn.Linear(self.features_count, 256)
        # self.fc2_m = nn.Linear(256, self.actions)
        self.fc2_m = nn.Linear(256, 200)

        self.apply(weights_init_orthogonal)

import math
import copy
import numpy as np
from random import randrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from adaptive_rl.mcts import MCTS, Node
import datetime

def selfplay(agent, model, output_list, first_move = False):
    #torch.cuda.set_device(1)
    # mem_states = torch.zeros((agent.actions*2, agent.games_in_iteration, 4, agent.env.height, agent.env.width), dtype=torch.int16, device = agent.device)
    # mem_policies = torch.zeros((agent.actions*2, agent.games_in_iteration, agent.actions), device=agent.device)
    agent.to()
    model.to(agent.device)
    #agent.t_one = torch.tensor([1],device=agent.device)
    #agent.env.t_one = torch.tensor([1], device=agent.device)

    max_steps = 200
    mem_states = torch.zeros((max_steps, agent.games_in_iteration, govars.NUM_CHNLS, agent.env.height, agent.env.width),
                             dtype=torch.int16, device=agent.device)
    mem_policies = torch.zeros((max_steps, agent.games_in_iteration, agent.actions), device=agent.device)

    game_indicies = torch.arange(agent.games_in_iteration)
    if first_move:
        states, moves = agent.env.first_move_states(agent.games_in_iteration)
        mcts_actions = 1
    else:
        states, moves = agent.env.zero_states(agent.games_in_iteration)
        mcts_actions = 0
    replay_buffer = Experience_buffer_GO(agent.actions * agent.games_in_iteration * 8 + 1, agent.env.height, agent.env.width, 0)

    mcts_list = [MCTS(agent.cpuct) for i in range(agent.games_in_iteration)]

    step = 0
    while True:
        if step == max_steps:
            print("ran out of steps")
            return
        mem_states[step] = states

        with torch.no_grad():
            actions, policies = agent.run_mcts(states, moves, model, mcts_list, mcts_actions, True)
        mem_policies[step] = policies

        states, rewards, moves, terminals = agent.env.step(actions, states)
        step += 1
        mcts_actions += 1

        end_game_indices = torch.nonzero(terminals)
        dim0, dim1 = end_game_indices.shape

        if dim0 != 0:
            for t_index in torch.flip(end_game_indices, [0]):
                index = t_index.item()
                mcts_list.pop(index)
                #print(states[index])
                replay_buffer.store(mem_states[0:step, index].view(-1, govars.NUM_CHNLS, agent.env.height, agent.env.width).float(), mem_policies[0:step, index].view(-1, agent.actions), rewards[index].item())

            non_terminals = torch.where(terminals == 0, agent.t_one, agent.t_zero)
            game_indicies = torch.nonzero(non_terminals)
            dim0, dim1 = game_indicies.shape

            if dim0 == 0:
                output_list.append((replay_buffer.index, replay_buffer.states[0:replay_buffer.index], replay_buffer.policies[0:replay_buffer.index], replay_buffer.values[0:replay_buffer.index], replay_buffer.moves_left[0:replay_buffer.index]))
                # print(actions)
                # print(policies)
                return

            game_indicies = game_indicies.view(-1)
            new_mem_states = torch.zeros((max_steps, dim0, govars.NUM_CHNLS, agent.env.height, agent.env.width), device = agent.device, dtype = torch.int16)
            new_mem_policies = torch.zeros((max_steps, dim0, agent.actions), device = agent.device)

            new_mem_states[0:step] = mem_states[0:step, game_indicies]
            new_mem_policies[0:step] = mem_policies[0:step, game_indicies]

            mem_states, mem_policies = new_mem_states, new_mem_policies
            states, moves = states[game_indicies], moves[game_indicies]

def arena_learning(agent, current_model, best_model, output_list, first_move = False):
    #torch.cuda.set_device(1)
    # mem_states = torch.zeros((agent.actions*2, agent.games_in_iteration, 4, agent.env.height, agent.env.width), dtype=torch.int16, device = agent.device)
    # mem_policies = torch.zeros((agent.actions*2, agent.games_in_iteration, agent.actions), device=agent.device)

    agent.to()
    current_model.to(agent.device)
    best_model.to(agent.device)

    max_steps = 200
    mem_states = torch.zeros((max_steps, agent.games_in_iteration, govars.NUM_CHNLS, agent.env.height, agent.env.width),
                             dtype=torch.int16, device=agent.device)
    mem_policies = torch.zeros((max_steps, agent.games_in_iteration, agent.actions), device=agent.device)

    game_indicies = torch.arange(agent.games_in_iteration)
    if first_move:
        states, moves = agent.env.first_move_states(agent.games_in_iteration)
        mcts_actions = 1
    else:
        states, moves = agent.env.zero_states(agent.games_in_iteration)
        mcts_actions = 0
    replay_buffer = Experience_buffer_GO(agent.actions * agent.games_in_iteration * 8 + 1, agent.env.height, agent.env.width, 0)

    mcts_list = [MCTS(agent.cpuct) for i in range(agent.games_in_iteration)]

    step = 0
    while True:
        if step == max_steps:
            print("ran out of steps")
            return
        mem_states[step] = states

        with torch.no_grad():
            actions, policies = agent.run_mcts(states, moves, current_model, mcts_list, mcts_actions, True)
        mem_policies[step] = policies

        states, rewards, moves, terminals = agent.env.step(actions, states)
        step += 1
        mcts_actions += 1

        end_game_indices = torch.nonzero(terminals)
        dim0, dim1 = end_game_indices.shape

        if dim0 != 0:
            for t_index in torch.flip(end_game_indices, [0]):
                index = t_index.item()
                mcts_list.pop(index)
                #print(states[index])
                replay_buffer.store(mem_states[0:step, index].view(-1, govars.NUM_CHNLS, agent.env.height, agent.env.width).float(), mem_policies[0:step, index].view(-1, agent.actions), rewards[index].item())

            non_terminals = torch.where(terminals == 0, agent.t_one, agent.t_zero)
            game_indicies = torch.nonzero(non_terminals)
            dim0, dim1 = game_indicies.shape

            if dim0 == 0:
                output_list.append((replay_buffer.index, replay_buffer.states[0:replay_buffer.index], replay_buffer.policies[0:replay_buffer.index], replay_buffer.values[0:replay_buffer.index], replay_buffer.moves_left[0:replay_buffer.index]))
                return

            game_indicies = game_indicies.view(-1)
            new_mem_states = torch.zeros((max_steps, dim0, govars.NUM_CHNLS, agent.env.height, agent.env.width), device = agent.device, dtype = torch.int16)
            new_mem_policies = torch.zeros((max_steps, dim0, agent.actions), device = agent.device)

            new_mem_states[0:step] = mem_states[0:step, game_indicies]
            new_mem_policies[0:step] = mem_policies[0:step, game_indicies]

            mem_states, mem_policies = new_mem_states, new_mem_policies
            states, moves = states[game_indicies], moves[game_indicies]

def arena(agent, model, indices, output_list):
    #torch.cuda.set_device(1)

    #agent.t_one = torch.tensor([1],device=agent.device)
    #agent.env.t_one = torch.tensor([1], device=agent.device)

    agent.to()
    model.to(agent.device)

    win, loss, draw = 0, 0, 0
    model2 = copy.deepcopy(model)
    model2.to(agent.device)
    for index in indices:
        # net.load_state_dict(torch.load('models/' + name + '_' + str(game) + '.pt'))
        model2.load_state_dict(torch.load('models/best_kaggle_model.pt'))

        for i in range(2):
            player1 = i % 2 == 0
            terminal = False

            mcts, mcts2 = MCTS(agent.cpuct), MCTS(agent.cpuct)
            state, move = agent.env.zero_states(1)
            step = 0

            while not terminal:
                with torch.no_grad():
                    if player1:
                        action, _ = agent.run_mcts(state, move, model, [mcts], step, False, False)
                    else:
                        action, _ = agent.run_mcts(state, move, model2, [mcts2], step, False, False)

                state, r, move, terminal = agent.env.step(action, state)
                step += 1

                if terminal[0] > 0:
                    print("game ended in: " + str(step) + " steps")
                    print("score: " + str(r))
                    if r > 0:
                        if player1:
                            win += 1
                        else:
                            loss += 1
                    elif r < 0:
                        if player1:
                            loss += 1
                        else:
                            win += 1
                    else:
                        draw += 1
                        break
                    print(state)

                player1 = not player1
    output_list.append((win, loss, draw))

def arena_training(agent, current_model, best_model, output_list, games = 5, player1 = True):
    #torch.cuda.set_device(1)

    agent.to()
    current_model.to(agent.device)
    best_model.to(agent.device)

    win, loss, draw = 0, 0, 0
    mcts1 = [MCTS(agent.cpuct) for i in range(games)]
    mcts2 = [MCTS(agent.cpuct) for i in range(games)]

    states, moves = agent.env.zero_states(games)
    step = 0

    starting_player1 = player1

    while True:
        if step > 300:
            print("arena ran over 300 steps")
            return
        with torch.no_grad():
            if player1:
                actions, _ = agent.run_mcts(states, moves, current_model, mcts1, step, True, False)
            else:
                actions, _ = agent.run_mcts(states, moves, best_model, mcts2, step, True, False)

        states, rewards, moves, terminals = agent.env.step(actions, states)
        step += 1

        end_game_indices = torch.nonzero(terminals)
        dim0, dim1 = end_game_indices.shape

        if dim0 != 0:
            for t_index in torch.flip(end_game_indices, [0]):
                index = t_index.item()
                mcts1.pop(index)
                mcts2.pop(index)
                if rewards[index] > 0:
                    if player1:
                        win += 1
                    else:
                        loss += 1
                elif rewards[index] < 0:
                    if player1:
                        loss += 1
                    else:
                        win += 1
                else:
                    draw += 1

            non_terminals = torch.where(terminals == 0, agent.t_one, agent.t_zero)
            game_indicies = torch.nonzero(non_terminals)
            dim0, dim1 = game_indicies.shape

            if dim0 == 0:
                output_list.append((win, loss, draw))
                return

            game_indicies = game_indicies.view(-1)
            states, moves = states[game_indicies], moves[game_indicies]

        player1 = not player1

class AZAgent:
    def __init__(self, env, device, simulation_count = 100, cpuct = 1.25, dirichlet_alpha = 0.15, exploration_fraction = 0.25,
                 name = 'azt', games_in_iteration = 200):
        #torch.cuda.set_device(1)
        self.env = env

        self.t_one = torch.tensor([1])
        self.t_zero = torch.tensor([0])
        self.device = device
        self.env.to(self.device)
        self.t_one = self.t_one.to(self.device)
        self.t_zero = self.t_zero.to(self.device)

        self.actions = self.env.height * self.env.width + 1
        self.simulation_count = simulation_count
        self.name = name
        self.cpuct = cpuct
        self.games_in_iteration = games_in_iteration

        self.dirichlet_alpha = dirichlet_alpha

        self.exploration_fraction = exploration_fraction
        self.exploration_fraction_inv = 1 - exploration_fraction

    def to(self, device=None):
        if device == None:
            device = self.device
        self.device = device
        self.t_one = self.t_one.to(device)
        self.t_zero = self.t_zero.to(device)
        self.env.to(self.device)

    def cpu(self):
        self.t_one = self.t_one.to(device="cpu")
        self.t_zero = self.t_zero.to(device="cpu")
        self.env.to(device="cpu")

    def run_mcts(self, states, moves, model, mcts_list, step, noise_b = True, training = True):
        length = len(mcts_list)
        # moves_length = self.actions - step
        # moves_length = -(states[:, 2].sum(2).sum(1)) + self.actions
        moves_length = moves.sum(1)


        mcts_states = torch.zeros((self.games_in_iteration, 6, self.env.height, self.env.width), device = self.device, dtype = torch.int16)
        mcts_actions = torch.zeros((self.games_in_iteration, 1), device = self.device, dtype = torch.long)
        mcts_indices = torch.zeros((self.games_in_iteration), dtype = torch.long)



        noise = [torch.from_numpy(np.random.dirichlet(np.ones(moves_length[i].short().item()) * self.dirichlet_alpha)) for i in range(moves_length.shape[0])]
        probs, values, _ = model(states[:,[govars.BLACK, govars.WHITE, govars.PASS_CHNL]].float())
        probs, values = F.softmax(probs, dim = 1), F.softmax(values, dim = 1)
        #print(probs)
        #print("------------")
        values = (torch.argmax(values, dim = 1) - 1).view(-1, 1)
        states, moves, probs, values = states.cpu(), moves.cpu(), probs.cpu(), values.cpu()
        #print(self.env.encode(states[0]))
        #print(probs)
        #print(values)

        index = 0
        for i in range(length):
            encode_state = self.env.encode(states[i])
            if encode_state in mcts_list[i].nodes:
                node = mcts_list[i].nodes[encode_state]
            else:
                node = Node(states[i], probs[i], values[i], moves[i], False)
                mcts_list[i].nodes[encode_state] = node

            if noise_b:
                node.P = node.P * self.exploration_fraction_inv + noise[i] * self.exploration_fraction
                node.P = node.P / node.P.sum()
            mcts_list[i].root = node

        for simulation in range(self.simulation_count):
            index = 0
            for i in range(length):
                mcts = mcts_list[i]
                mcts.selection()

                if mcts.current_node is not None:
                    mcts.backup(mcts.current_node, mcts.parents)
                else:
                    node, action_index = mcts.parents[0]
                    mcts_states[index] = node.state
                    mcts_actions[index, 0] = node.moves[action_index, 0]
                    mcts_indices[index] = i
                    index += 1

            if index > 0:
                states, rewards, moves, terminals = self.env.step(mcts_actions[0:index], mcts_states[0:index])
                probs, values, _ = model(states[:, [govars.BLACK, govars.WHITE, govars.PASS_CHNL]].float())
                probs, values = F.softmax(probs, dim = 1), F.softmax(values, dim = 1)
                values = (torch.argmax(values, dim = 1) - 1).view(-1, 1)
                states, moves, probs, values, rewards, terminals = states.cpu(), moves.cpu(), probs.cpu(), values.cpu(), rewards.cpu(), terminals.cpu()

                for i in range(index):
                    mcts_index = mcts_indices[i]
                    mcts = mcts_list[mcts_index]
                    parent, action_index = mcts.parents[0]
                    if terminals[i] > 0:
                        node = Node(states[i], probs[i], - rewards[i], moves[i], True)
                        parent.children[action_index] = node
                    else:
                        encode_state = self.env.encode(states[i])
                        if encode_state in mcts.nodes:
                            node = mcts.nodes[encode_state]
                        else:
                            node = Node(states[i], probs[i], values[i], moves[i], False)
                            mcts.nodes[encode_state] = node
                            parent.children[action_index] = node
                    mcts.backup(node, mcts.parents)

        policy_list = []
        for i in range(length):
            policy_list.append(mcts_list[i].root.getProbs(self.actions))
        policies = torch.stack(policy_list)
        #(policies)
        if training:
            actions = policies.multinomial(num_samples = 1)
        else:
            actions = torch.argmax(policies, dim = 1).view(-1, 1)
        return actions.to(self.device), policies.to(self.device)
