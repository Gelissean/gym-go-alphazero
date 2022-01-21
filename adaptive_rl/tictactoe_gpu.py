import torch

class TicTacToe:
    def __init__(self, width = 5, height = 5, win_count = 4):
        self.width = width
        self.height = height
        self.win_count = win_count
        self.max_moves = width * height
        self.state_size = self.max_moves * 2

        possible_win = []

        for y in range(self.height):
            for x in range(self.width):
                horizontal = x + self.win_count <= self.width
                vertical = y + self.win_count <= self.height

                if horizontal:
                    t = torch.zeros((self.height, self.width), dtype = torch.int16)
                    for i in range(self.win_count):
                        t[y, x + i] = 1
                    possible_win.append(t)
                if vertical:
                    t = torch.zeros((self.height, self.width), dtype = torch.int16)
                    for i in range(self.win_count):
                        t[y + i, x] = 1
                    possible_win.append(t)
                if horizontal and vertical:
                    t = torch.zeros((self.height, self.width), dtype = torch.int16)
                    for i in range(self.win_count):
                        t[y + i, x + i] = 1
                    possible_win.append(t)
                if horizontal and y - self.win_count + 1 >= 0:
                    t = torch.zeros((self.height, self.width), dtype = torch.int16)
                    for i in range(self.win_count):
                        t[y - i, x + i] = 1
                    possible_win.append(t)

        self.check_kernels_length = len(possible_win)
        self.check_kernels = torch.stack(possible_win).view(1, -1, self.height, self.width)
        self.device = ''

        self.t_one = torch.tensor([1])
        self.t_zero = torch.tensor([0])

    def to(self, device):
        self.check_kernels = self.check_kernels.to(device)
        self.device = device
        self.t_one = self.t_one.to(device)
        self.t_zero = self.t_zero.to(device)

    def zero_states(self, count = 1):
        return torch.zeros((count, 2, self.height, self.width), dtype=torch.int16, device = self.device), torch.ones((count, self.max_moves), device = self.device, dtype = torch.long) #Bx1x5x5, Bx25

    def first_move_states(self, count = 1):
        rep = count // self.max_moves
        mod = count % self.max_moves
        order = torch.arange(count, device = self.device)
        indices = torch.arange(self.max_moves, device = self.device).repeat(rep)
        if mod != 0:
            indices = torch.cat((indices, torch.arange(mod, device = self.device)), 0)
        states_indices = order * self.state_size + indices + self.max_moves
        states, moves = torch.zeros((count, 2, self.height, self.width), dtype=torch.int16, device = self.device).view(-1), torch.ones((count, self.max_moves), device = self.device, dtype = torch.long).view(-1) #Bx1x5x5, Bx25
        states[states_indices] = 1
        moves_indices = order * self.max_moves + indices
        moves[moves_indices] = 0
        return states.view(-1, 2, self.height, self.width), moves.view(-1, self.max_moves)

    def possible_moves(self, states):
        states_sum = states.sum(1)
        return torch.where(states_sum == 0, self.t_one, self.t_zero).view(-1, self.max_moves).long() #Bx5x5

    def step(self, actions, states): #actions Bx1, states Bx2x5x5
        dim1, dim2 = actions.shape
        order = torch.arange(dim1, device = self.device)
        indices = order * self.state_size + actions.view(-1)
        states = states.clone().view(-1) #B*5*5
        states[indices] = 1
        states = states.view(-1, 2, self.height, self.width) #Bx2x5x5

        states_current_player = states[:, 0].clone().view(-1, 1, self.height, self.width) #Bx1x5x5
        current_player = torch.where(states_current_player == 1, self.t_one, self.t_zero).repeat(1, self.check_kernels_length, 1, 1) #Bx28x5x5

        result = current_player * self.check_kernels #Bx28x5x5
        area_sum = result.sum((2, 3)) #Bx28
        win_sum = torch.where(area_sum == self.win_count, self.t_one, self.t_zero).sum(1) #B

        moves = self.possible_moves(states)
        moves_sum = moves.sum((1))
        no_moves = torch.where(moves_sum == 0, self.t_one, self.t_zero)
        terminals = torch.where(win_sum + no_moves > 0, self.t_one, self.t_zero)
        
        states[:, 0] = states[:, 1]
        states[:, 1] = states_current_player.view(-1, self.height, self.width)


        return states, win_sum.float(), moves, terminals #states, rewards, moves, terminals

    def show_board(self, state, co = 'O', cx = 'X'):
        s = ''
        for y in range(self.height):
            for x in range(self.width):
                if state[0, y, x] == 1:
                    s += ' ' + cx + ' '
                elif state[1, y, x] == 1:
                    s += ' ' + co + ' '
                else:
                    s += '   '
            s += '\n'
        return s

    def check_win(self, state, x, y):
        x_start = max(x - self.win_count, 0)
        x_end = min(x + self.win_count, self.width)

        y_start = max(y - self.win_count, 0)
        y_end = min(y + self.win_count, self.height)

        sum = 0
        for i in range(x_start, x_end):
            if state[y, i] == 1:
                sum += 1
                if sum == self.win_count:
                    return True
            else:
                sum = 0

        sum = 0
        for i in range(y_start, y_end):
            if state[i, x] == 1:
                sum += 1
                if sum == self.win_count:
                    return True
            else:
                sum = 0

        d_start = min(x - x_start, y - y_start)
        d_end = min(x_end - x, y_end - y)
        d_x, d_y = x - d_start, y - d_start
        d_range = x + d_end - d_x

        sum = 0
        for i in range(d_range):
            if state[d_y + i, d_x + i] == 1:
                sum += 1
                if sum == self.win_count:
                    return True
            else:
                sum = 0

        d_start = min(x_end - x - 1, y - y_start)
        d_end = min(x - x_start + 1, y_end - y)
        d_x, d_y = x + d_start, y - d_start
        d_range = y + d_end - d_y

        sum = 0
        for i in range(d_range):
            if state[d_y + i, d_x - i] == 1:
                sum += 1
                if sum == self.win_count:
                    return True
            else:
                sum = 0

        return False

    def encode(self, state):
        code = ''
        for y in range(self.height):
            for x in range(self.width):
                if state[0, y, x] == 1:
                    code += 'x'
                elif state[1, y, x] == 1:
                    code += 'o'
                else:
                    code += ' '
        return code
