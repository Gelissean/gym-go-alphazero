import argparse

import gym
import torch

from adaptive_rl.mcts import MCTS
from alphazero_modded import AZAgent, Net_GO

from environment.Env_Go import Env_Go

# Arguments
parser = argparse.ArgumentParser(description='Demo Go Environment')
parser.add_argument('--boardsize', type=int, default=6)
parser.add_argument('--komi', type=float, default=0)
args = parser.parse_args()

# Initialize environment
go_env = gym.make('gym_go:go-v0', size=args.boardsize, komi=args.komi, reward_method="real")
state = go_env.reset()

env = Env_Go()

print(state)

# Game loop
done = False
i = 0
player = True

model = Net_GO(4, 7, 64, 6 * 6, 37)

model.load_state_dict(torch.load("saves/subor.state_dict"))
#model.load_state_dict(torch.load("models/go_6_0_3_8.pt"))
#model.load_state_dict(torch.load('models/go_6_0_1_6.pt'))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device: ', device)
model.to(device)

agent = AZAgent(env, device="cuda", games_in_iteration=1, simulation_count=100, name="AlphaZero_GO")

mcts_list = [MCTS(agent.cpuct) for i in range(agent.games_in_iteration)]


def make_states_and_moves(state, env):
    all_states = torch.tensor(state, device="cuda")
    if state[2, 0, 0] == 0:
        states = all_states[[0, 1, 3, 4]]
    else:
        states = all_states[[1, 0, 3, 4]]
    states = states.reshape(1, 4, 6, 6)
    # moves = states[1, 2]
    moves = torch.where(states[0, 2]==0, torch.tensor(1, device="cuda"), torch.tensor(0,device="cuda"))
    moves = torch.cat((moves.reshape(-1), torch.tensor([1], device="cuda")), 0).reshape(1, -1)
    return states, moves


while not done:
    if player:
        done = False
        while not done:
            action = go_env.render(mode="human")
            if action is None:
                done =1
            else:
                x, y = action
                if state[3, x,y] == 0:
                    done = 1
    else:
        states, moves = make_states_and_moves(state, go_env)
        mcts_actions = i
        actions, policies = agent.run_mcts(states, moves, model, mcts_list, mcts_actions, False)
        action = actions[0].item()
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
