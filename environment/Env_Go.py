from abc import ABC

from environment.Environment import Environment

import torch
import gym
from utils import change_state_and_run
from gym_go import gogame
import numpy


class Env_Go(Environment, ABC):
    def __init__(self, size=6, komi=0, reward_method="real", device="cpu"):
        self.size = size
        self.width = size
        self.height = size
        self.max_moves = size * size + 1
        self.state_size = 4 * (size * size)

        self.komi = komi
        self.reward_method = reward_method

        self.device = device
        self.t_one = torch.tensor([1], device=self.device)
        self.t_zero = torch.tensor([0], device=self.device)
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
        return torch.zeros((count, 4, self.size, self.size), dtype=torch.int16, device = self.device) ,\
               torch.ones((count, self.max_moves), device = self.device, dtype = torch.long) #Bx4x6x6, Bx37

    # returns groups of states as if the player was starting second
    def first_move_states(self, count):
        state_size = self.max_moves - 1
        rep = count // state_size
        mod = count % state_size
        order = torch.arange(count, device=self.device)
        indices = torch.arange(state_size, device=self.device).repeat(rep)
        if mod != 0:
            indices = torch.cat((indices, torch.arange(mod, device=self.device)), 0)
        states_indices = order * (3 * state_size) + indices + state_size
        states_indices2 = order * (3 * state_size) + indices + state_size*2
        states = torch.zeros((count, 3, self.size, self.size), dtype=torch.int16, device=self.device).view(-1) # Bx1xHxW
        moves = torch.ones((count, self.max_moves), dtype=torch.long, device=self.device).view(-1)  # Bx(WxH+1)
        states[states_indices] = 1
        states[states_indices2] = 1
        moves_indices = order * self.max_moves + indices
        moves[moves_indices] = 0
        return torch.cat(((states.view(count, 3, self.size, self.size)),
                          torch.zeros(count, 1, self.size, self.size, device=self.device)), 1),\
               moves.view(-1, self.max_moves)

    def possible_moves(self, states):
        legal_moves = torch.where(states[:, 2] == 0, self.t_one, self.t_zero).view(-1, self.max_moves - 1).long()
        pass_legality = torch.ones(states.shape[0], device=self.device, dtype=torch.int16).view(-1, 1)
        return torch.cat((legal_moves, pass_legality), 1)

    def step(self, actions, states):

        zeros = torch.zeros((states.shape[0]), 1, self.height, self.width, device=self.device, dtype=torch.int16)
        temp_states = torch.cat((states[:, :3], zeros, states[:, 3:], zeros), 1)

        batch_states = gogame.batch_next_states(temp_states.detach().cpu().numpy(), actions.reshape(-1).detach().cpu().numpy())

        new_states = torch.from_numpy(numpy.delete(batch_states, [2, 3, 5],1))
        new_states = new_states.to(self.device)
        batch_gameover = gogame.batch_game_ended(batch_states)
        batch_rewards = gogame.batch_winning(batch_states, self.komi)

        rewards = batch_gameover*batch_rewards
        rewards = torch.tensor(rewards, device=self.device)

        moves = gogame.batch_valid_moves(batch_states)

        invalid_move_count = numpy.sum(numpy.sum(batch_states[:, 3], 2), 1)
        pass_legality = numpy.where(invalid_move_count > 22, 1, 0)
        moves[:, -1] = pass_legality

        moves = torch.from_numpy(moves)
        moves.to(self.device)

        terminals = torch.tensor(batch_gameover, device=self.device)

        mask = numpy.where(batch_states[:, 2, 0, 0] == 1, 1, 0)
        if mask.sum() > 0:
            new_states = self._flip_states(new_states, numpy.where(mask==1))
            #new_states = torch.cat((new_states[:, 1].reshape(-1, 1, self.size, self.size), new_states[:, [0, 2]]), 1)

        #moves = self.possible_moves(states)
        return new_states, rewards, moves, terminals  # states, rewards, moves, terminals


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
        code = []
        for y in range(self.size):
            for x in range(self.size):
                if state[0, y, x] == 1:
                    code.append('B')
                elif state[1, y, x] == 1:
                    code.append('W')
                else:
                    code.append(' ')

        for y in range(self.size):
            for x in range(self.size):
                if state[2, y, x] == 1:
                    code.append('l')
                else:
                    code.append(' ')
        if state[3,0,0]:
            code.append('x')
        else:
            code.append(' ')
        return ''.join(code)

    def _init_boards(self, count):
        self.envs = []
        self.rewards = torch.zeros(count)
        for i in range(count):
            env = gym.make('gym_go:go-v0', size=self.size, komi=self.komi, reward_method=self.reward_method)
            env.reset()
            self.envs.append(env)

    def _flip_states(self, states, mask):
        c = states[mask[0], 0].detach().clone()
        d = states[mask[0], 1].detach().clone()
        e = states
        e[mask[0], 0] = d
        e[mask[0], 1] = c
        return states
