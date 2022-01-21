# adaptive-rl
 
Welcome dear guest!

Our repository is focused on implementation of AlphaZero algorithm extended by adaptive method for AlphaZero. In our case, training iteration consists of collecting data by playing parallel games, training current model and arena. We use module of arena, because we didn't have a lof of computation power druing the training and we were not able to collect data and traing model asynchronously. On the other hand, we implemented fully GPU environment of Tic-Tac-Toe.

One of adaptive method is written in file mcts_a.py. All adaptive results are published in paper (link is comming soon).

You can run script game.py or game_a.py and play against one of trained models. There are few trained models in "models" directory. 

This  work  was  supported  by  grant  VEGA  1/0089/19  andGrant System of University ofˇZilina No. 1/2020 (8041).
