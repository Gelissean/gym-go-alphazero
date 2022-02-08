from abc import ABC

from environment.Environment import Environment

import torch
import gym
from utils import change_state_and_run


class Env_Go(Environment, ABC):
    def __init__(self, size=6, komi=0, reward_method="heuristic"):
        self.size = size
        self.width = size
        self.height = size
        self.max_moves = size * size + 1
        self.state_size = 4 * (size * size)

        self.komi = komi
        self.reward_method = reward_method

        self.t_one = torch.tensor([1])
        self.t_zero = torch.tensor([0])
        self.envs = []
        self.rewards = []
        # self.check_kernels_length = len(possible_win)
        # self.check_kernels = torch.stack(possible_win).view(1, -1, self.height, self.width)

    def to(self, device):
        # self.check_kernels = self.check_kernels.to(device)
        self.device = device
        self.t_one = self.t_one.to(device)
        self.t_zero = self.t_zero.to(device)

    #returns state at the beginning of the game
    def zero_states(self, count):
        self._init_boards(count)
        return torch.zeros((count, 4, self.size, self.size), dtype=torch.int16, device = self.device) ,\
               torch.ones((count, self.max_moves), device = self.device, dtype = torch.long) #Bx4x6x6, Bx37

    # returns groups of states as if the player was starting second
    def first_move_states(self, count):
        self._init_boards(count)
        state_size = self.max_moves - 1
        rep = count // state_size
        mod = count % state_size
        order = torch.arange(count)
        indices = torch.arange(state_size).repeat(rep)
        if mod != 0:
            indices = torch.cat((indices, torch.arange(mod)), 0)
        states_indices = order * (3 * state_size) + indices + state_size
        states_indices2 = order * (3 * state_size) + indices + state_size*2
        states = torch.zeros((count, 3, self.size, self.size), dtype=torch.int16).view(-1) # Bx1xHxW
        moves = torch.ones((count, self.max_moves), dtype=torch.long).view(-1)  # Bx(WxH+1)
        states[states_indices] = 1
        states[states_indices2] = 1
        moves_indices = order * self.max_moves + indices
        moves[moves_indices] = 0
        return torch.cat(((states.view(count, 3, self.size, self.size)),
                          torch.ones(count, 1, self.size, self.size)), 1),\
               moves.view(-1, self.max_moves)

    def possible_moves(self, states):
        legal_moves = torch.where(states[:, 2] == 0, self.t_one, self.t_zero).view(-1, self.max_moves - 1).long()
        pass_legality = torch.ones(states.shape[0]).view(-1, 1)
        return torch.cat((legal_moves, pass_legality), 1)

    def step(self, actions, states):
        self._init_boards(states.shape[0])
        possible_moves = self.possible_moves(states)
        for i in range(len(self.envs)):
            states[i], self.rewards[i], moves, terminals = \
                change_state_and_run(self.envs[i],
                                     states[i],
                                     actions[i],
                                     possible_moves[i],
                                     self.size, self.size)
        moves = self.possible_moves(states)
        terminals = torch.where(states[:, 3, 0, 0] == 1, self.t_one, self.t_zero)
        # for i in range(terminals.shape[0]):
        #     if terminals[i].item() == 1:
        #         self.envs[i].reset()
        return states, self.rewards, moves, terminals  # states, rewards, moves, terminals


    def show_board(self, state, cx='B', co='W'):
        s = ''
        for y in range(self.size):
            for x in range(self.size):
                if state[0, y, x] == 1:
                    s += ' ' + cx + ' '
                elif state[1, y, x] == 1:
                    s += ' ' + co + ' '
                else:
                    s += '   '
            s += '\n'
        return s

    # TODO
    def check_win(self, state, x, y):
        super().check_win()

    def encode(self, state):
        code = ''
        for y in range(self.size):
            for x in range(self.size):
                if state[0, y, x] == 1:
                    code += 'B'
                elif state[1, y, x] == 1:
                    code += 'W'
                else:
                    code += ' '
        return code

    def _init_boards(self, count):
        self.envs = []
        self.rewards = torch.zeros(count)
        for i in range(count):
            env = gym.make('gym_go:go-v0', size=self.size, komi=self.komi, reward_method=self.reward_method)
            env.reset()
            self.envs.append(env)
