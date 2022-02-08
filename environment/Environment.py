class Environment:
    def __init__(self):
        raise NotImplementedError

    def to(self, device):
        raise NotImplementedError

    def zero_states(self, count):
        raise NotImplementedError

    def first_move_states(self, count):
        raise NotImplementedError

    def possible_moves(self, states):
        raise NotImplementedError

    def step(self, actions, states):
        raise NotImplementedError

    def show_board(self, state, co, cx):
        raise NotImplementedError

    def check_win(self, state, x, y):
        raise NotImplementedError

    def encode(self, state):
        raise NotImplementedError
