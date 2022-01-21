import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from tictactoe_gpu import TicTacToe
from alphazero import AZAgent, arena
from replay_buffer import ExperienceReplay
from net import Net
import datetime
from mcts_a import MCTS

if __name__ == '__main__':
    width = 6
    height = 6
    win_count = 4
    actions = width * height
    beta = 10
    name = 'azt_a6_6_7'
    
    torch.cuda.set_device(1)

    env = TicTacToe(width = width, height = height, win_count = win_count)

    net = Net(2, 7, 64, actions)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device: ', device)
    net.to(device)
    net.load_state_dict(torch.load('models/' + name + '_363.pt'))

    agent = AZAgent(env, device = device, name = name)
    play_game = True
    while play_game:
	    answer = input('Do you want to take first move? (y/n)')
	    first_move = answer == 'y'
	    MCTS(agent.cpuct, beta, actions)
	    
	    if first_move:
	    	computer = False
	    	players_mark = 'X'
	    	computers_mark = 'O'
	    	print('You are player X')
	    else:
	    	computer = True
	    	players_mark = 'O'
	    	computers_mark = 'X'
	    	print('You are player O')

	    state = env.zero_states()
	    terminal = False
	    while not terminal:
	    	if computer:
	    		action, _ = agent.run_mcts(state, move, model, [mcts], step, False, False)
	    	else:
	    		input_x = int(input('Write X index of cell'))
	    		input_y = int(input('Write Y index of cell'))
	    		action = torch.tensor([[input_y * height + input_x]]).to(device).long()


	    	state, reward, moves, terminals = agent.env.step(action, state)

	    	if computer:
	    		show_board_with_labels(state[0], co = computers_mark, cx = players_mark)
	    	terminal = terminals[0].item()

	    	if terminal:
	    		if not computer:
	    			show_board_with_labels(state[0], co = players_mark, cx = computers_mark)
	    		if reward[0] > 0.5:
	    			if computer:
	    				print('End game, computer is winner!')
	    			else:
	    				print('End game, you are winner!')
	    		else:
	    			print('End game, it is draw!')

	    		answer = input('Do you want to play again? (y/n)')
	    		play_game = answer == 'y'
	    		if play_game:
	    			print('/n/--------------------------------------------n')


def show_board_with_labels(self, state, co = 'O', cx = 'X'):
	s = '   '
    _, _, height, width = state.shape
    for i in range(width):
    	s += ' i '

    for y in range(height):
        for x in range(width):
            if state[0, y, x] == 1:
                s += ' ' + cx + ' '
            elif state[1, y, x] == 1:
                s += ' ' + co + ' '
            else:
                s += '   '
        s += '\n y '
    return s