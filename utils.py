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

def encode(state: torch.tensor):
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

def change_state_and_run(env, state, action, possible_moves, height=6, width=6, device="cpu"):
    if possible_moves[action].item() == 1:
        tempstate = state.reshape(4, height, width).cpu()
        temp = numpy.array([0 for i in range(env.env.size*env.env.size)]).reshape(1,env.env.size, env.env.size)
        temp = numpy.concatenate((tempstate[0:2].numpy(), temp, tempstate[2:].numpy(), temp), 0)
        env.env.state_ = temp
        env.env.done = False
        states, reward, done, info = env.step(action.item())
        moves = torch.tensor(numpy.array([swap_moves(states[3])]), device=device)
        states = torch.tensor(numpy.array([change_state(states)]), device=device)
        return states, torch.tensor([reward], device=device), moves, torch.tensor([done], device=device)
    else:
        state[3] = state[3] + 1
        reward = torch.tensor(1, device=device)
        reward[0] = -10
        return state, reward, possible_moves, torch.tensor([True], device=device)
