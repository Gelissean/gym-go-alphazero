from adaptive_rl.net import Net, ResidualBlock, weights_init_xavier
from adaptive_rl.replay_buffer import ExperienceReplay
import torch

class Experience_buffer_GO(ExperienceReplay):
    def __init__(self, size, height, width, batch_size):
        super().__init__(size, height, width, batch_size)
        self.actions = height * width + 1
        self.states = torch.zeros((size, 4, height, width), dtype=torch.int16)
        self.policies = torch.zeros((size, self.actions))

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
        self.fc2_m = nn.Linear(256, self.actions)

        self.apply(weights_init_xavier)

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
    mem_states = torch.zeros((agent.actions, agent.games_in_iteration, 4, agent.env.height, agent.env.width), dtype=torch.int16, device = agent.device)
    mem_policies = torch.zeros((agent.actions, agent.games_in_iteration, agent.actions), device=agent.device)

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
                replay_buffer.store(mem_states[0:step, index].view(-1, 2, agent.env.height, agent.env.width).float(), mem_policies[0:step, index].view(-1, agent.actions), rewards[index].item())

            non_terminals = torch.where(terminals == 0, agent.t_one, agent.t_zero)
            game_indicies = torch.nonzero(non_terminals)
            dim0, dim1 = game_indicies.shape

            if dim0 == 0:
                output_list.append((replay_buffer.index, replay_buffer.states[0:replay_buffer.index], replay_buffer.policies[0:replay_buffer.index], replay_buffer.values[0:replay_buffer.index], replay_buffer.moves_left[0:replay_buffer.index]))
                return

            game_indicies = game_indicies.view(-1)
            new_mem_states = torch.zeros((agent.actions, dim0, 2, agent.env.height, agent.env.width), device = agent.device, dtype = torch.int16)
            new_mem_policies = torch.zeros((agent.actions, dim0, agent.actions), device = agent.device)

            new_mem_states[0:step] = mem_states[0:step, game_indicies]
            new_mem_policies[0:step] = mem_policies[0:step, game_indicies]

            mem_states, mem_policies = new_mem_states, new_mem_policies
            states, moves = states[game_indicies], moves[game_indicies]

def arena(agent, model, indices, output_list):
    #torch.cuda.set_device(1)
    win, loss, draw = 0, 0, 0
    model2 = copy.deepcopy(model)
    model2.to(agent.device)
    for index in indices:
        model2.load_state_dict(torch.load('models/' + agent.name + '_' + str(index.item()) + '.pt'))

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
                    if r > 0:
                        if player1:
                            win += 1
                        else:
                            loss += 1
                    else:
                        draw += 1
                    break

                player1 = not player1
    output_list.append((win, loss, draw))

def arena_training(agent, current_model, best_model, output_list, games = 5, player1 = True):
    #torch.cuda.set_device(1)
    win, loss, draw = 0, 0, 0
    mcts1 = [MCTS(agent.cpuct) for i in range(games)]
    mcts2 = [MCTS(agent.cpuct) for i in range(games)]

    states, moves = agent.env.zero_states(games)
    step = 0

    while True:
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

        self.actions = self.env.height * self.env.width
        self.simulation_count = simulation_count
        self.name = name
        self.cpuct = cpuct
        self.games_in_iteration = games_in_iteration

        self.dirichlet_alpha = dirichlet_alpha

        self.exploration_fraction = exploration_fraction
        self.exploration_fraction_inv = 1 - exploration_fraction

    def run_mcts(self, states, moves, model, mcts_list, step, noise_b = True, training = True):
        length = len(mcts_list)
        moves_length = self.actions - step

        mcts_states = torch.zeros((self.games_in_iteration, 2, self.env.height, self.env.width), device = self.device, dtype = torch.int16)
        mcts_actions = torch.zeros((self.games_in_iteration, 1), device = self.device, dtype = torch.long)
        mcts_indices = torch.zeros((self.games_in_iteration), dtype = torch.long)

        noise = torch.from_numpy(np.random.dirichlet(np.ones(moves_length) * self.dirichlet_alpha, length))
        probs, values, _ = model(states.float())
        probs, values = F.softmax(probs, dim = 1), F.softmax(values, dim = 1)
        values = (torch.argmax(values, dim = 1) - 1).view(-1, 1)
        states, moves, probs, values = states.cpu(), moves.cpu(), probs.cpu(), values.cpu()

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
                probs, values, _ = model(states.float())
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
        if training:
            actions = policies.multinomial(num_samples = 1)
        else:
            actions = torch.argmax(policies, dim = 1).view(-1, 1)
        return actions.to(self.device), policies.to(self.device)
