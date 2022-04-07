import argparse

import gym
import torch
import numpy

from adaptive_rl.mcts import MCTS
from alphazero_modded import AZAgent, Net_GO

from environment.Env_Go import Env_Go
from gym_go import govars

# Arguments
parser = argparse.ArgumentParser(description='Demo Go Environment')
parser.add_argument('--boardsize', type=int, default=6)
parser.add_argument('--komi', type=float, default=0)
args = parser.parse_args()

# Initialize environment
go_env = gym.make('gym_go:go-v0', size=args.boardsize, komi=args.komi, reward_method="real")
state = go_env.reset()

env = Env_Go()

#print(state)

# Game loop
done = False
i = 0
player = False

model = Net_GO(3, 7, 64, 6 * 6, 37)

#model.load_state_dict(torch.load("models/go_6_3vrstvy_0_0.pt"))
#model.load_state_dict(torch.load("models/go_6_3vrstvy_kaggle_0_0.pt"))
#model.load_state_dict(torch.load("models/go_6_3vrstvy_kaggle_1_2.pt"))
model.load_state_dict(torch.load("saves/subor_3_vrstvy_go_6_3vrstvy_high_LR_3_13.state_dict"))
#model.load_state_dict(torch.load("saves/subor_3_vrstvy_restart_current_19.state_dict"))
#model.load_state_dict(torch.load("models/go_6_3vrstvy_5_3.pt"))
#model.load_state_dict(torch.load("saves/subor_3_vrstvy_restart.state_dict"))
#model.load_state_dict(torch.load("models/go_6_3vrstvy_restart_from_kaggle_2_0.pt"))
#model.load_state_dict(torch.load('models/go_6_0_1_6.pt'))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device: ', device)
model.to(device)

agent = AZAgent(env, device="cuda", games_in_iteration=1, simulation_count=1000, name="AlphaZero_GO")

mcts_list = [MCTS(agent.cpuct) for i in range(agent.games_in_iteration)]

def flip_states(states, policies):
    policies = policies[:,:-1].view(-1, 1, 6,6)
    print(policies)
    print(torch.flip(policies, [3]))

    for i in range(3):
        policies = torch.rot90(policies, 1, [2, 3])
        print(policies)
        print(torch.flip(policies, [3]))

    states = states
    print(env.show_board(states[0]))
    print(env.show_board(torch.flip(states, [3])[0]))

    for i in range(3):
        states = torch.rot90(states, 1, [2, 3])
        print(env.show_board(states[0]))
        print(env.show_board(torch.flip(states, [3])[0]))

def make_states_and_moves(state, env):
    all_states = torch.tensor(state, device="cuda")
    if state[2, 0, 0] == 0:
        states = all_states[[0, 1, 2, 3, 4, 5]]
    else:
        states = all_states[[1, 0, 2, 3, 4, 5]]
    states = states.reshape(1, 6, 6, 6)
    # moves = states[1, 2]
    moves = torch.where(states[0, 3]==0, torch.tensor(1, device="cuda"), torch.tensor(0,device="cuda"))
    #print(states)

    invalid_move_count = numpy.sum(numpy.sum(state[3], 1), 0)
    if invalid_move_count > 22:
        value = 1
    else:
        value = 0

    moves = torch.cat((moves.reshape(-1), torch.tensor([value], device="cuda")), 0).reshape(1, -1)
    return states, moves


while not done:
    if player:
        done = False
        while not done:
            action = go_env.render(mode="human")
            if action is None:
                done = 1
            else:
                x, y = action
                if state[govars.INVD_CHNL, x,y] == 0:
                    done = 1
    else:
        states, moves = make_states_and_moves(state, go_env)
        mcts_actions = i
        actions, policies = agent.run_mcts(states, moves, model, mcts_list, mcts_actions, False, False)
        action = actions[0].item()
        action = policies.argmax().item()
        # flip_states(states, policies)
        print(policies)
    #action = 36
    player = not player
    #action = go_env.uniform_random_action()
    state, reward, done, info = go_env.step(action)
    #continue;

    #print(action)
    #print(state)
    print(reward)
    i += 1
    #print(done)
    #print(info)

    if go_env.game_ended():
        break
go_env.render(mode="human")
print(i)
print(state)
