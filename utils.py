import numpy
import torch


def swap_moves(moves: numpy.ndarray):
    return numpy.append((moves + 1) % 2, 1) #otocit mozne akcie na 1, pridat pass


def change_state(state: numpy.ndarray, normalise=True):
    if state[2, 0, 0] == 1:
        temp = state[0].copy()
        state[0] = state[1]
        state[1] = temp
    return numpy.delete(state, [2, 5], 0)  # remove player turn layer and game over layer

def encode(state: torch.Tensor):
    code = ''
    for y in range(state.size(1)):
        for x in range(state.size(2)):
            if state[0, y, x] == 1:
                code += 'x'
            elif state[1, y, x] == 1:
                code += 'o'
            else:
                code += ' '
    return code

def change_state_and_run(env, state, action):
    tempstate = state.reshape(4,6,6)
    temp = numpy.array([0 for i in range(env.env.size*env.env.size)]).reshape(1,env.env.size, env.env.size)
    temp = numpy.concatenate((tempstate[0:2].numpy(), temp, tempstate[2:].numpy(), temp), 0)
    env.env.state_ = temp
    env.env.done = False
    states, reward, done, info = env.step(action.item())
    moves = torch.FloatTensor(numpy.array([swap_moves(states[3])]))
    states = torch.FloatTensor(numpy.array([change_state(states)]))
    return states, torch.Tensor([reward]), moves, torch.Tensor([done])
