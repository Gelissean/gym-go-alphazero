import torch
import numpy as np

print(torch.from_numpy(np.random.dirichlet(np.ones(10) * 0.01, 5)))




'''
if current_index > 5:
    print('arena')
    start = datetime.datetime.now()
    with Manager() as manager:
        output_list = manager.list()
        processes = []
        for i in range(2):
            p = Process(target=arena, args=(agent, net, current_index - 1, output_list,))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    win, lose, draw = 0, 0, 0
    for i in output_list:
        if i == 1:
            win += 1
        elif i == -1:
            lose += 1
        else:
            draw += 1

    sum = win + lose
    win_rate = float(win) / sum
    print('win:', win, 'lose:', lose, 'draw:', draw, 'win rate:', win_rate)
    if sum > 0 and win_rate >= 0.55:
        torch.save(self.model.state_dict(), 'models/' + agent.name + '_' + str(current_index) + '.pt')
        print('save', current_index)
        current_index += 1
    else:
        model2.load_state_dict(torch.load('models/' + agent.name + '_' + str(current_index - 1) + '.pt'))
    print(datetime.datetime.now() - start)
else:
    torch.save(self.model.state_dict(), 'models/' + agent.name + '_' + str(current_index) + '.pt')
    print('save', current_index)
    current_index += 1
'''
