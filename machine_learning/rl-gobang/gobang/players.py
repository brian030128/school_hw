import numpy as np
from mcts import MCTS

class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())
        return a


class HumanGobangPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board, player):
        valid = self.game.getValidMoves(board, player)
        #for i in range(len(valid)):
        #    if valid[i]:
        #        print(int(i/self.game.n), int(i%self.game.n))
        while True:
            a = input().strip()

            x,y = [int(x) for x in a.split(' ')]
            a = self.game.n * x + y if x!= -1 else self.game.n ** 2
            if valid[a]:
                break
            else:
                print('Invalid')

        return a


import argparse
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_episodes', type=int)
    parser.add_argument('--num_iterations', type=int)
    parser.add_argument('--keep_iters', type=int, default=20)
    parser.add_argument('--pk_episodes', type=int, default=40)
    parser.add_argument('--num_mcts_sims', type=int, default=1000)
    parser.add_argument('--cpuct', type=int, default=1)
    parser.add_argument("--seed", type=int, default=524126, help="Random seed for reproduction")
    args = parser.parse_args()
    return args

class AlphaZeroPlayer():
    def __init__(self, game, nnet, args=get_args()):
        self.game = game
        self.nnet = nnet
        self.mcts = MCTS(self.game, self.nnet, args)


    def play(self, board, player, temp=0):
        board = self.game.getCanonicalForm(board, player)
        pi = self.mcts.getActionProb(board, temp=temp)
        action = np.random.choice(len(pi), p=pi)
        return action
