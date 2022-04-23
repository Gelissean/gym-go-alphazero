import argparse
import os
import shutil

import gym
import torch
import numpy

from adaptive_rl.mcts import MCTS
from adapted_mcts_a import MCTS as MCTS_adaptible
from alphazero_modded import AZAgent, Net_GO
from adaptible_agent import AZAgent_Adaptible
from data_exporter import data_exporter

from environment.Env_Go import Env_Go
from time import time
from gym_go import govars, gogame
import copy
import csv

#first_model = "models/go_6_3vrstvy_high_LR_7_batches_exp_15.pt"
first_model = "models/most_trained_best.pt"
second_model = 'models/go_6_3vrstvy_high_LR_7_batches_exp_5.pt'
use_noise = False
max_steps = 86


def arena(model_A, agent_A, model_B, agent_B):
    # torch.cuda.set_device(1)

    # agent.t_one = torch.tensor([1],device=agent.device)
    # agent.env.t_one = torch.tensor([1], device=agent.device)

    agent_A.to()
    model_A.to(agent_A.device)
    agent_B.to()
    model_B.to(agent_B.device)

    results = []
    player1 = govars.BLACK
    for i in range(2):
        win, loss, draw = 0, 0, 0
        starting_p = player1
        terminal = False

        mcts, mcts2 = MCTS(agent_A.cpuct), MCTS(agent_B.cpuct)
        state, move = agent_A.env.zero_states(1)
        step = 0

        while not terminal:
            with torch.no_grad():
                if player1 == govars.WHITE:
                    action, _ = agent_B.run_mcts(state, move, model_B, [mcts2], step, use_noise, False)
                else:
                    action, _ = agent_A.run_mcts(state, move, model_A, [mcts], step, use_noise, False)

            state, r, move, terminal = agent_A.env.step(action.detach().cpu().numpy().reshape(-1), state)
            step += 1

            if step > 200:
                print("Game took too long, evaluating manually")
                barea, warea = gogame.areas(state[0])
                difference = abs(barea - warea)
                if difference > 2:
                    if barea > warea:
                        r = 1
                    else:
                        r = -1
                else:
                    r = 0
                terminal = [True]

            if terminal[0] > 0:
                print("game ended in: " + str(step) + " steps")
                if starting_p == govars.WHITE:
                    r = -r
                print("score: " + str(r))
                if r > 0:
                    win += 1
                elif r < 0:
                    loss += 1
                else:
                    draw += 1
                if starting_p == govars.BLACK:
                    x = 'A'
                    o = 'B'
                else:
                    x = 'B'
                    o = 'A'
                print(agent_A.env.show_board(state[0], cx=x, co=o))
                results.append((win, loss, draw, step))

            player1 = not player1
    return results


def single_arena(model_A_path, model_B_path):
    env = Env_Go()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent_A = AZAgent(env, device=device, games_in_iteration=1, simulation_count=400, name="AlphaZero_GO")
    agent_B = AZAgent(env, device=device, games_in_iteration=1, simulation_count=400, name="AlphaZero_GO_other")

    model_A = Net_GO(3, 7, 64, 6 * 6, 37)
    model_A.load_state_dict(torch.load(model_A_path, map_location=device))
    model_A.to(agent_A.device)

    model_B = copy.deepcopy(model_A)
    model_A.load_state_dict(torch.load(model_B_path, map_location=device))
    model_B.to(agent_B.device)

    start_time = time()
    results = arena(model_A, agent_A, model_B, agent_B)
    print(results, time() - start_time)
    return results


def folder_arena(folder_path, writer):
    files = os.listdir(folder_path)
    if writer is not None:
        writer.write(["starting player", "steps", "win", "loss", "draw"])
    for file in files:
        wx = 0
        file_path = folder_path + file
        print(file_path)
        results = single_arena(first_model, file_path)
        i = 0
        for result in results:
            w, l, d, s = result
            wx = wx + w
            if writer is not None:
                writer.write([i, s, w, l, d])
            i = i + 1
        if wx == 2:
            shutil.copyfile(file_path, "models/experiments-sorted/easy/" + file)
        elif wx == 0:
            shutil.copyfile(file_path, "models/experiments-sorted/hard/" + file)
        else:
            shutil.copyfile(file_path, "models/experiments-sorted/medium/" + file)


def arena_adaptible(model_A, agent_A, model_B, agent_B, beta):
    # torch.cuda.set_device(1)

    # agent.t_one = torch.tensor([1],device=agent.device)
    # agent.env.t_one = torch.tensor([1], device=agent.device)

    agent_A.to()
    model_A.to(agent_A.device)
    agent_B.to()
    model_B.to(agent_B.device)

    win, loss, draw = 0, 0, 0
    results = []
    player1 = govars.BLACK
    for i in range(2):
        starting_p = player1
        terminal = False

        mcts, mcts2 = MCTS_adaptible(agent_A.cpuct, beta, max_steps), MCTS(agent_B.cpuct)
        state, move = agent_A.env.zero_states(1)
        step = 0

        while not terminal:
            with torch.no_grad():
                if player1 == govars.WHITE:
                    action, _ = agent_B.run_mcts(state, move, model_B, [mcts2], step, use_noise, False)
                else:
                    action, _ = agent_A.run_mcts(state, move, model_A, [mcts], step, use_noise, False)

            state, r, move, terminal = agent_A.env.step(action.detach().cpu().numpy().reshape(-1), state)
            step += 1

            if step > 200:
                print("Game took too long, evaluating manually")
                barea, warea = gogame.areas(state[0])
                difference = abs(barea - warea)
                if difference > 2:
                    if barea > warea:
                        r = 1
                    else:
                        r = -1
                else:
                    r = 0
                terminal = [True]

            if terminal[0] > 0:
                print("game ended in: " + str(step) + " steps")
                if starting_p == govars.WHITE:
                    r = -r
                print("score: " + str(r))
                if r > 0:
                    win += 1
                elif r < 0:
                    loss += 1
                else:
                    draw += 1
                    break
                if starting_p == govars.BLACK:
                    x = 'A'
                    o = 'B'
                else:
                    x = 'B'
                    o = 'A'
                print(agent_A.env.show_board(state[0], cx=x, co=o))
                results.append((win, loss, draw, step))

            player1 = not player1
    return results


def single_arena_adapt(model_A_path, model_B_path, beta, writer):
    env = Env_Go()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent_A = AZAgent_Adaptible(env, device=device, games_in_iteration=1, simulation_count=400,
                                name="AlphaZero_GO_adaptible", max_steps=max_steps)
    agent_B = AZAgent(env, device=device, games_in_iteration=1, simulation_count=400, name="AlphaZero_GO_other")

    model_A = Net_GO(3, 7, 64, 6 * 6, 37)
    model_A.load_state_dict(torch.load(model_A_path, map_location=device))
    model_A.to(agent_A.device)

    model_B = copy.deepcopy(model_A)
    model_A.load_state_dict(torch.load(model_B_path, map_location=device))
    model_B.to(agent_B.device)

    start_time = time()
    results = arena_adaptible(model_A, agent_A, model_B, agent_B, beta)
    print(results, time() - start_time)
    if writer is not None:
        i = 0
        for result in results:
            w, l, d, s = result
            writer.write([str(i), str(beta), str(s), str(w), str(l), str(d), model_B_path])
            i = i + 1
    return


def folder_arena_adapt(folder_path, beta_vals, result_path_name):
    files = os.listdir(folder_path)
    with data_exporter(result_path_name) as writer:
        for beta in beta_vals:
            print("starting with beta:", beta)
            #i = 0
            for file in files:
                #i = i + 1
                #if i > 9:
                    #break
                file_path = folder_path + file
                print(file_path)
                writer.write(["starting player", "beta", "steps", "win", "loss", "draw", "model_name"])
                single_arena_adapt(first_model, file_path, beta, writer)

def main():
    #single_arena(first_model, second_model)
    #with data_exporter("data.csv") as exporter:
        #folder_arena("models/experiments-unsorted/", exporter)
    beta_values = [0, 1, 2, 3, 4, 5, 7.5, 10]
    folder_arena_adapt("results/easy/", beta_values, "results/easy_adaptible.csv")
    #single_arena_adapt(first_model, second_model, 0, None)
    #single_arena_adapt(first_model, second_model, 5, None)

if __name__ == "__main__":
    main()