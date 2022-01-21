import torch
import math

class Node:
    def __init__(self, state, probs, value, moves, terminal = False):
        self.state = state
        self.moves = torch.nonzero(moves)
        self.P = probs[self.moves].view(-1)
        self.P = self.P / self.P.sum()
        self.value = value
        length, _ = self.moves.shape
        self.length = length

        self.N = torch.zeros(length, dtype = torch.int32)
        self.Q = torch.zeros(length)
        self.T = terminal

        self.children = {}

    def getProbs(self, size):
        probs = self.N.float() / self.N.sum()
        all_probs = torch.zeros(size)
        all_probs[self.moves.view(-1)] = probs
        return all_probs

class MCTS:
    def __init__(self, cpuct):
        self.cpuct = cpuct
        self.nodes = {}

        self.root = None
        self.current_node = None
        self.parents = []

    def set_parents(self, parents):
        self.parents = parents

    def selection(self):
        game_indicies = torch.nonzero(self.root.N == 0)
        for index in game_indicies:
            self.parents = [(self.root, index)]
            self.current_node = None
            return

        parents = []
        node = self.root

        while True:
            if node.T == True:
                parents.reverse()
                self.parents = parents
                self.current_node = node
                return

            N_sum = node.N.sum().item()
            sq = math.sqrt(float(N_sum))

            if N_sum > 0:
                u = node.Q + self.cpuct * node.P * sq / (1.0 + node.N)
                index = torch.argmax(u).item()
            else:
                index = torch.argmax(node.P).item()

            parents.append((node, index))

            if index in node.children:
                node = node.children[index]
            else:
                parents.reverse()
                self.parents = parents
                self.current_node = None
                return

    def backup(self, node, parents):
        v = node.value

        for parent, i in parents:
            v = - v
            count = parent.N[i] + 1
            parent.Q[i] = (parent.N[i] * parent.Q[i] + v) / count
            parent.N[i] = count
