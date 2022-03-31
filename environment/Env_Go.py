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
        moves = torch.ones((count, self.max_moves), device = self.device, dtype = torch.long)
        moves[:, -1] = 0
        return torch.zeros((count, govars.NUM_CHNLS, self.size, self.size), dtype=torch.int16, device = self.device) ,\
               moves #Bx4x6x6, Bx37

    # returns groups of states as if the player was starting second
    def first_move_states(self, count):
        states, moves = self.zero_states(count)
        states = gogame.batch_next_states(states.detach().cpu().numpy(), numpy.arange(count)%(self.size*self.size))
        return torch.tensor(states, device=self.device, dtype=torch.int16).reshape(-1, govars.NUM_CHNLS, self.size, self.size),\
               torch.tensor(gogame.batch_valid_moves(states), device=self.device, dtype=torch.int16)

    def possible_moves(self, states):
        legal_moves = torch.where(states[:, govars.INVD_CHNL] == 0, self.t_one, self.t_zero).view(-1, self.max_moves - 1).long()
        pass_legality = torch.ones(states.shape[0], device=self.device, dtype=torch.int16).view(-1, 1)
        return torch.cat((legal_moves, pass_legality), 1)

    def step(self, actions, states):
        batch_states = gogame.batch_next_states(states.detach().cpu().numpy(), actions.reshape(-1).detach().cpu().numpy())

        new_states = torch.tensor(batch_states, device=self.device, dtype=torch.int16)
        batch_gameover = gogame.batch_game_ended(batch_states)
        batch_rewards = gogame.batch_winning(batch_states, self.komi)

        rewards = batch_gameover*batch_rewards
        rewards = torch.tensor(rewards, device=self.device)

        moves = gogame.batch_valid_moves(batch_states)

        invalid_move_count = numpy.sum(numpy.sum(batch_states[:, govars.INVD_CHNL], 2), 1)
        pass_legality = numpy.where(invalid_move_count > 22, 1, 0)
        moves[:, -1] = pass_legality

        moves = torch.from_numpy(moves)
        moves = moves.short()
        moves.to(self.device)

        terminals = torch.tensor(batch_gameover, device=self.device)

        mask = numpy.where(batch_states[:, govars.TURN_CHNL, 0, 0] == 1, 1, 0)
        if mask.sum() > 0:
            new_states = self._flip_states(new_states, numpy.where(mask==1))
            #new_states = torch.cat((new_states[:, 1].reshape(-1, 1, self.size, self.size), new_states[:, [0, 2]]), 1)

        #moves = self.possible_moves(states)
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
