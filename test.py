from alphazero_modded import AZAgent
from environment.Env_Go import Env_Go


class A():
    def funkcia(self):
        print("som v A")

class B():
    def funkcia(self):
        print("som v B")

class C(A):
    def funkcia(self):
        super(C, self).funkcia()
        print("som v C")

class D(A,B):
    def daco(self):
        print("haha daco")

class F(B, A):
    def daco(self):
        print("haha daco")

class E(A,B):
    def funkcia(self):
        print("som v E")

if __name__=="__main__":
    a = A()
    b = B()
    c = C()
    d = D()
    e = E()
    f = F()

    print("A")
    a.funkcia()
    print("B")
    b.funkcia()
    print("C")
    c.funkcia()
    print("D")
    d.funkcia()
    print("E")
    e.funkcia()
    print("F")
    f.funkcia()


def f(hodnota, b):
    print(hodnota)
    #print(b)
    b.to()
    print(b.t_one)

import numpy, torch, torch.multiprocessing as mp
#from GymGo_main import main
#main()
if __name__ == "__main__":
    env = Env_Go()
    mp.set_start_method('spawn')
    for i in range(2):
        for j in range(2):
            if i == 0:
                x = torch.rand(5, device="cuda")
            else:
                x = torch.rand(5, device="cpu")
            x.share_memory_()
            if j == 0:
                agent = AZAgent(env, device="cuda", games_in_iteration=1, simulation_count=100, name="AlphaZero_GO")
            else:
                agent = AZAgent(env, device="cpu", games_in_iteration=1, simulation_count=100, name="AlphaZero_GO")

            agent.cpu()
            p = mp.Process(target=f, args=(x, agent))
            p.start()
            p.join()
    print("main done")
