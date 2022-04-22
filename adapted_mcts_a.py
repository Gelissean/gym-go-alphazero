import torch
import math

class Node:
    def __init__(self, state, probs, value, length, moves, max_steps, terminal = False):
        self.state = state
        self.moves = torch.nonzero(moves)
        self.P = probs[self.moves].view(-1)
        self.P = self.P / self.P.sum()
        self.value = value
        self.length = length
        size, _ = self.moves.shape
        self.size = size

        self.N = torch.zeros(size, dtype = torch.int32)
        self.Q = torch.zeros(size)
        self.L = torch.zeros(size)
        self.T = terminal
        self.max_steps = max_steps

        self.children = {}

    def getProbs(self, size):
        probs = self.N.float() / self.N.sum()
        all_probs = torch.zeros(size)
        all_probs[self.moves.view(-1)] = probs
        return all_probs

class MCTS:
    def __init__(self, cpuct, beta):
        self.cpuct = cpuct
        self.beta = beta
        self.nodes = {}

        self.root = None
        self.current_node = None
        self.parents = []

    def set_parents(self, parents):
        self.parents = parents

    def selection(self, step):
        game_indicies = torch.nonzero(self.root.N == 0)
        for index in game_indicies:
            self.parents = [(self.root, index.item())]
            self.current_node = None
            return

        parents = []
        node = self.root
        best_player = True

        while True:
            if node.T == True:
                parents.reverse()
                self.parents = parents
                self.current_node = node
                return

            N_sum = node.N.sum().item()
            sq = math.sqrt(float(N_sum))

            if best_player:
                alpha = step / node.max_steps
                if N_sum > 0:
                    b = node.Q + self.cpuct * node.P * sq / (1.0 + node.N)
                    c = node.Q + self.beta * node.L 
                    u = alpha * b + (1 - alpha) * c
                    index = torch.argmax(u).item()
                else:
                    index = torch.argmax(node.P).item()
            else:
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
            step += 1

    def backup(self, node, parents):
        v = node.value
        l = node.length

        for parent, i in parents:
            v = - v
            count = parent.N[i] + 1
            parent.Q[i] = (parent.N[i] * parent.Q[i] + v) / count
            parent.L[i] = (parent.N[i] * parent.L[i] + l) / count
            parent.N[i] = count
            l -= 1
