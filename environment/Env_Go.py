from abc import ABC

from environment.Environment import Environment

import torch
import gym
from utils import change_state_and_run
from gym_go import gogame, govars
import numpy


class Env_Go(Environment, ABC):
    def __init__(self, size=6, komi=0, reward_method="real", device="cpu"):
        self.size = size
        self.width = size
        self.height = size
        self.max_moves = size * size + 1
        self.state_size = govars.NUM_CHNLS * (size * size)

        self.moves_before_pass = 22

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
        states = gogame.batch_init_state(count, self.size)
        moves = gogame.batch_valid_moves(states)
        moves[:, -1] = 0 # cant pass on start
        return states, moves

    # returns groups of states as if the player was starting second
    def first_move_states(self, count):
        states, moves = self.zero_states(count)
        states = gogame.batch_next_states(states, numpy.arange(count)%(self.size*self.size))
        moves = gogame.batch_valid_moves(states)
        moves[:, -1] = 0 # cant pass on second turn
        return states, moves

    def possible_moves(self, states):
        moves = gogame.batch_valid_moves(states)
        moves[:,-1] = numpy.where((numpy.sum(moves,1))<(self.max_moves - self.moves_before_pass), 1, 0)
        return moves

    def step(self, actions, states):
        new_states = gogame.batch_next_states(states, actions)
        terminals = gogame.batch_game_ended(new_states)
        batch_rewards = gogame.batch_winning(new_states, self.komi)
        rewards = terminals*batch_rewards
        moves = self.possible_moves(new_states)

        moves = moves.short()
        return new_states, rewards, moves, terminals  # states, rewards, moves, terminals


    def show_board(self, state, cx='B', co='W'):
        s = ''
        for y in range(self.size):
            for x in range(self.size):
                if state[govars.BLACK, y, x] == 1:
                    s += ' ' + cx + ' '
                elif state[govars.WHITE, y, x] == 1:
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
                if state[govars.BLACK, y, x] == 1:
                    code.append('B')
                elif state[govars.WHITE, y, x] == 1:
                    code.append('W')
                else:
                    code.append(' ')

        for y in range(self.size):
            for x in range(self.size):
                if state[govars.INVD_CHNL, y, x] == 1:
                    code.append('l')
                else:
                    code.append(' ')
        if state[govars.PASS_CHNL,0,0]:
            code.append('x')
        else:
            code.append(' ')
        return ''.join(code)

    def _flip_states(self, states, mask):
        temp = states[mask]
        temp[:, [0, 1, 2]] = temp[:, [1, 0, 2]]
        states[mask] = temp
        return states

    def process_states(self, states):
        processed_states = torch.tensor(states[:, [govars.BLACK, govars.WHITE, govars.PASS_CHNL]], device=self.device,dtype=torch.int16)

        mask = numpy.argwhere(states[:, govars.TURN_CHNL, 0, 0]==govars.WHITE).reshape(-1)
        processed_states = self._flip_states(processed_states, mask)
        return processed_states
