import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from torch.multiprocessing import Manager, Process, set_start_method
from tictactoe_gpu import TicTacToe
from alphazero import AZAgent, arena
from replay_buffer import ExperienceReplay
from net import Net
import datetime

if __name__ == '__main__':
    width = 5
    height = 5
    win_count = 4
    actions = width * height

    cpus = 9
    name = 'azt_a'
    games_count = 54
    
    torch.cuda.set_device(1)

    set_start_method('spawn')

    env = TicTacToe(width = width, height = height, win_count = win_count)

    net = Net(2, 5, 64, actions)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device: ', device)
    net.to(device)

    agent = AZAgent(env, device = device, name = name)
    agent.dirichlet_alpha_min = 0.01
    
    indices = torch.tensor(range(games_count))
    interval = games_count // cpus
    
    for game in range(40, games_count + 1):
        #start_time = datetime.datetime.now()
        net.load_state_dict(torch.load('models/' + name + '_' + str(game) + '.pt'))
        
        win, loss, draw = 0, 0, 0
        with Manager() as manager:
            output_list = manager.list()
            processes = []
            for i in range(cpus):
                p = Process(target=arena, args=(agent, net, indices[i*interval:(i+1)*interval], output_list))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
    
            for i in range(len(output_list)):
                w, l, d = output_list[i]
                win += w
                loss += l 
                draw += d
        
        print('game: ', game, 'win: ', win, 'loss: ', loss, 'draw: ', draw)
        #print(datetime.datetime.now() - start_time)
            

