import numpy as np
import torch

class ExperienceReplay:
    def __init__(self, size, height, width, batch_size):
        self.states = torch.zeros((size, 2, height, width), dtype = torch.int16)
        self.policies = torch.zeros((size, height * width))
        self.values = torch.zeros((size, 1), dtype = torch.long)
        self.moves_left = torch.zeros((size, 1), dtype = torch.long)

        self.max_size = size
        self.index = 0
        self.batch_size = batch_size

        self.full_buffer = False

        self.height = height
        self.width = width
        self.actions = height * width

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

        self.add(length, states, policies, values, moves)

        policies = policies.view(-1, 1, self.height, self.width)
        self.add(length, torch.flip(states, [3]), torch.flip(policies, [3]).view(-1, self.actions), values, moves)

        for i in range(3):
            states, policies = torch.rot90(states, 1, [2, 3]), torch.rot90(policies, 1, [2, 3])
            self.add(length, states, policies.reshape(-1, self.actions), values, moves)
            self.add(length, torch.flip(states, [3]), torch.flip(policies, [3]).reshape(-1, self.actions), values, moves)

    def sample(self):
        if self.full_buffer:
            indices = np.random.choice(self.max_size, self.batch_size)
        else:
            indices = np.random.choice(self.index, self.batch_size)
        return self.states[indices], self.policies[indices], self.values[indices], self.moves_left[indices]
