from gobang.game import GobangGame
from gobang.players import AlphaZeroPlayer
from gobang.min_max_player import LeveledMinMaxPlayer
from net import NeuralNet
import torch
import time
import copy
import multiprocessing as mp


class MultiThreadedArena:
    def __init__(self, game, threads=10):
        self.game = game
        self.threads = threads
        self.results = []
        self.manager = mp.Manager()
    
    def pk(self, player1, player2, num_games=100):
        results = self.manager.list()
        games_started = mp.Value('i', 0)
        processes = []

        for i in range(self.threads):
            p = mp.Process(target=worker, args=(self.game, player1, player2, num_games, games_started, results))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()

        return results

def worker(game, player1, player2, target_games, started_games, results):
    while True:
        with started_games.get_lock():
            if started_games.value >= target_games:
                break
            started_games.value += 2
        p1 = copy.deepcopy(player1)
        p2 = copy.deepcopy(player2)
        result1 = play_single_game (game, p1, p2)
        results.append(result1)

        p1 = copy.deepcopy(player1)
        p2 = copy.deepcopy(player2)
        result2 = play_single_game(game, p2, p1)
        results.append(result2 * -1)

def play_single_game(game: GobangGame, player1, player2):
    board = game.getInitBoard()
    player = 1
    while True:
        action = player1.play(board, player) if player == 1 else player2.play(board, player)
        board, player = game.getNextState(board, player, action)
        result = game.getGameEnded(board, player)
        if result != 0:
            return result
    
import argparse
if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    arena = MultiThreadedArena(GobangGame(), threads=15)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    game = GobangGame()

    nn1 = NeuralNet(game).to(device)
    nn1.load_state_dict(torch.load("best.pth"))
    nn1.eval()
    nn2 = NeuralNet(game).to(device)
    nn2.eval()

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_episodes', type=int)
    parser.add_argument('--num_iterations', type=int)
    parser.add_argument('--keep_iters', type=int, default=20)
    parser.add_argument('--pk_episodes', type=int, default=40)
    parser.add_argument('--num_mcts_sims', type=int, default=25)
    parser.add_argument('--cpuct', type=int, default=1)
    args = parser.parse_args()


    player1 = AlphaZeroPlayer(game, nn1, copy.deepcopy(args))
    player2 = LeveledMinMaxPlayer(game, 0)
    #player2 = AlphaZeroPlayer(game, nn2, copy.deepcopy(args))

    start = time.time()
    result = arena.pk(player1, player2)
    print(result)
    print(len(result))
    print(result.count(1)/len(result))
    print("Total time taken:", time.time() - start)
