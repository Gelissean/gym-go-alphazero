import argparse

import gym

# Arguments
parser = argparse.ArgumentParser(description='Demo Go Environment')
parser.add_argument('--boardsize', type=int, default=6)
parser.add_argument('--komi', type=float, default=0)
args = parser.parse_args()

# Initialize environment
go_env = gym.make('gym_go:go-v0', size=args.boardsize, komi=args.komi, reward_method="heuristic")
state = go_env.reset()

# Game loop
done = False
i = 0
while not done:
    i += 1
    #action = go_env.render(mode="human")
    action = 36
    #action = go_env.uniform_random_action()
    state, reward, done, info = go_env.step(action)
    continue;

    #print(action)
    #print(state)
    print(reward)
    #print(done)
    #print(info)

    if go_env.game_ended():
        break
    action = go_env.uniform_random_action()
    state, reward, done, info = go_env.step(action)

    print("hrac white")

    #print(action)
    #print(state)
    print(reward)
    #print(done)
    #print(info)
    go_env.render(mode="human")
go_env.render(mode="human")
print(i)
