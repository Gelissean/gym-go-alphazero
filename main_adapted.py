import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from torch.multiprocessing import Manager, Process, set_start_method
from alphazero_modded import AZAgent, selfplay, arena_training, Net_GO, Experience_buffer_GO
from adaptive_rl.replay_buffer import ExperienceReplay
from adaptive_rl.net import Net
import datetime

from environment.Env_Go import Env_Go
from gym_go import govars

if __name__ == '__main__':
    games_in_iteration = 36

    board_size = 6
    komi = 0
    replay_buffer_size = 50000
    actions = board_size * board_size + 1
    iteration_count = 1000
    weight_decay = 0.0001
    lr = 0.01
    lrs = [lr, 0.001, 0.001, 0.0001, 0.0001]
    dirichlet_alphas = [0.3, 0.3, 0.15, 0.15, 0.03]

    cross_entropy = nn.CrossEntropyLoss()
    cross_entropy_moves = nn.CrossEntropyLoss()

    cpus = 2  # 6
    batch_size = 50*games_in_iteration #50*cpus*games_in_iteration
    batches_in_iteration = 10 #games_in_iteration*cpus
    name = 'go_6_3vrstvy_high_LR_6_batches_exp'

    torch.cuda.set_device(0)

    set_start_method('spawn')

    env = Env_Go()

    current_model = Net_GO(3, 7, 64, board_size*board_size, actions)

    load = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device: ', device)
    if load:
        current_model.load_state_dict(torch.load("saves/subor_3_vrstvy_go_6_3vrstvy_high_LR_3_13.state_dict"))
        current_model.eval()

        best_model = copy.deepcopy(current_model)
        #best_model.load_state_dict(torch.load("models/go_6_3vrstvy_restart_from_kaggle_2_5.pt"))
        best_model.load_state_dict(torch.load("saves/subor_3_vrstvy_go_6_3vrstvy_high_LR_3_13.state_dict"))
        best_model.eval()

    else:
        best_model = copy.deepcopy(current_model)
    current_model.to(device)
    best_model.to(device)

    #net.share_memory()
    optimizer = optim.Adam(current_model.parameters(), lr=lrs[0], weight_decay = weight_decay)




    agent = AZAgent(env, device = device, games_in_iteration = games_in_iteration, simulation_count = 400, name = name)
    replay_buffer = Experience_buffer_GO(replay_buffer_size, board_size, board_size, batch_size)

    #torch.load(best_model.state_dict(), "saves/subor.state_dict")

    model_index = 0
    for iteration in range(iteration_count):
        start = datetime.datetime.now()
        if iteration % 100 == 0:
            i = iteration // 100
            if i < (len(lrs)):
                for g in optimizer.param_groups:
                    g['lr'] = lrs[i]
                agent.dirichlet_alpha = dirichlet_alphas[i]
        print(start)
        print('selfplay', iteration)

        with Manager() as manager:
            output_list = manager.list()
            processes = []
            agent.cpu()
            best_model.to("cpu")
            for i in range(cpus):
                p = Process(target=selfplay, args=(agent, best_model, output_list, i % 2 == 1))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()

            for i in range(len(output_list)):
                length, states, policies, values, moves_left = output_list[i]
                replay_buffer.add(length, states, policies, values, moves_left)
            agent.to(device)
            best_model.to(device)

        print('index', replay_buffer.index)

        avg_loss_policy, avg_loss_value, avg_loss_moves = 0, 0, 0
        for b in range(batches_in_iteration):
            optimizer.zero_grad()
            states, target_policies, target_values, target_moves = replay_buffer.sample()
            model_policies, model_values, model_left_moves = current_model(states[:,[govars.BLACK,govars.WHITE,govars.PASS_CHNL]].to(device).float())
            loss_policy = - (target_policies.to(device) * F.log_softmax(model_policies, 1)).mean()
            loss_value = cross_entropy(model_values, target_values.view(-1).to(device))
            loss_left_moves = cross_entropy_moves(model_left_moves, target_moves.view(-1).to(device) )
            loss = loss_value + loss_policy + loss_left_moves
            avg_loss_policy += loss_policy.detach().item()
            avg_loss_value += loss_value.detach().item()
            avg_loss_moves += loss_left_moves.detach().item()
            loss.backward()
            optimizer.step()

        print('policy loss: ', (avg_loss_policy / batches_in_iteration), 'value loss: ', (avg_loss_value / batches_in_iteration), 'left moves loss: ', (avg_loss_moves / batches_in_iteration))
        print('update', (datetime.datetime.now() - start))



        #torch.cuda.empty_cache()
        #print(datetime.datetime.now() - start)
        #torch.save(best_model.state_dict(), "saves/subor_3_vrstvy_high_lr.state_dict")
        #torch.save(current_model.state_dict(), "saves/subor_3_vrstvy_"+ name + "_" + str(iteration) + ".state_dict")
        #continue


        torch.cuda.empty_cache()

        print('arena')
        win, loss, draw = 0, 0, 0
        with Manager() as manager:
            output_list = manager.list()
            processes = []
            agent.cpu()
            best_model.to("cpu")
            #current_model.to("cpu")
            for i in range(cpus):  # // 2):
                p = Process(target=arena_training, args=(agent, copy.deepcopy(current_model).to("cpu"), best_model, output_list, min(10, games_in_iteration), i % 2 == 0))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()

            for i in range(len(output_list)):
                w, l, d = output_list[i]
                win += w
                loss += l
                draw += d
            agent.to(device)
            best_model.to(device)
            #current_model.to(device)

        a_sum = win + loss
        if a_sum > 0 and win / float(a_sum) > 0.55:
            torch.save(current_model.state_dict(), 'models/' + name + '_' + str(model_index) + '.pt')
            best_model = copy.deepcopy(current_model)
            best_model.to(device)

            print('model is updated ', model_index)
            model_index += 1

        print('win: ', win, 'loss: ', loss, 'draw: ', draw)

        torch.cuda.empty_cache()
        print(datetime.datetime.now() - start)
        torch.save(best_model.state_dict(), "saves/subor_3_vrstvy_high_lr.state_dict")
        torch.save(current_model.state_dict(), "saves/subor_3_vrstvy_"+ name + "_" + str(iteration) + ".state_dict")