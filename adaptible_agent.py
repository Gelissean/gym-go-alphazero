import math
import copy
import numpy as np
from random import randrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from adapted_mcts_a import MCTS, Node
import datetime

class AZAgent_Adaptible:
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
        moves = torch.tensor(moves, dtype=torch.int16)
        moves_length = moves.sum(1)


        #mcts_states = torch.zeros((self.games_in_iteration, 6, self.env.height, self.env.width), device = self.device, dtype = torch.int16)
        mcts_states = np.zeros((self.games_in_iteration, 6, self.env.height, self.env.width))
        mcts_actions = torch.zeros((self.games_in_iteration, 1), device = self.device, dtype = torch.long)
        mcts_indices = torch.zeros((self.games_in_iteration), dtype = torch.long)



        noise = [torch.from_numpy(np.random.dirichlet(np.ones(moves_length[i].short().item()) * self.dirichlet_alpha)) for i in range(moves_length.shape[0])]
        probs, values, expected_length = model(self.env.process_states(states).float())
        probs, values, expected_length = F.softmax(probs, dim = 1), F.softmax(values, dim = 1), F.softmax(expected_length, dim = 1)
        #print(probs)
        #print("------------")
        values = (torch.argmax(values, dim = 1) - 1).view(-1, 1)
        expected_length = torch.argmax(expected_length, dim = 1)
        #states, moves, probs, values = states.cpu(), moves.cpu(), probs.cpu(), values.cpu()
        probs, values, expected_length = probs.cpu(), values.cpu(), expected_length.cpu()
        #print(self.env.encode(states[0]))
        #print(probs)
        #print(values)

        index = 0
        for i in range(length):
            encode_state = self.env.encode(states[i])
            if encode_state in mcts_list[i].nodes:
                node = mcts_list[i].nodes[encode_state]
            else:
                node = Node(states[i], probs[i], values[i], expected_length[i], moves[i], step + self.env.get_expected_moves(states[i]), False)
                mcts_list[i].nodes[encode_state] = node

            if noise_b:
                node.P = node.P * self.exploration_fraction_inv + noise[i] * self.exploration_fraction
                node.P = node.P / node.P.sum()
            mcts_list[i].root = node

        for simulation in range(self.simulation_count):
            index = 0
            for i in range(length):
                mcts = mcts_list[i]
                mcts.selection(step)

                if mcts.current_node is not None:
                    mcts.backup(mcts.current_node, mcts.parents)
                else:
                    node, action_index = mcts.parents[0]
                    mcts_states[index] = node.state
                    mcts_actions[index, 0] = node.moves[action_index, 0]
                    mcts_indices[index] = i
                    index += 1

            if index > 0:
                states, rewards, moves, terminals = self.env.step(mcts_actions[0:index].detach().cpu().numpy().reshape(-1), mcts_states[0:index])
                probs, values, expected_length = model(self.env.process_states(states).float())
                probs, values, expected_length = F.softmax(probs, dim = 1), F.softmax(values, dim = 1), F.softmax(expected_length, dim = 1)
                values = (torch.argmax(values, dim = 1) - 1).view(-1, 1)
                expected_length = torch.argmax(expected_length, dim = 1)
                moves, probs, values, expected_length = torch.tensor(moves, dtype=torch.int16), probs.cpu(), values.cpu(), expected_length.cpu()

                for i in range(index):
                    mcts_index = mcts_indices[i]
                    mcts = mcts_list[mcts_index]
                    parent, action_index = mcts.parents[0]
                    if terminals[i] > 0:
                        node = Node(states[i], probs[i], - rewards[i], expected_length[i], moves[i], step + self.env.get_expected_moves(states[i]), True)
                        parent.children[action_index] = node
                    else:
                        encode_state = self.env.encode(states[i])
                        if encode_state in mcts.nodes:
                            node = mcts.nodes[encode_state]
                        else:
                            node = Node(states[i], probs[i], values[i], expected_length[i], moves[i], step + self.env.get_expected_moves(states[i]), False)
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
