import threading
from gobang.game import GobangGame
from gobang.players import AlphaZeroPlayer
from net import NeuralNet
import torch
import time

def play_single_game(game_id):
    game = GobangGame()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 每個執行緒有自己的模型與玩家實例
    nn1 = NeuralNet(game).to(device)
    nn1.eval()
    nn2 = NeuralNet(game).to(device)
    nn2.eval()
    
    player1 = AlphaZeroPlayer(game, nn1)
    player2 = AlphaZeroPlayer(game, nn2)

    board = game.getInitBoard()
    player = 1

    while True:
        action = player1.play(board) if player == 1 else player2.play(board)
        board, player = game.getNextState(board, player, action)
        result = game.getGameEnded(board, player)
        if result != 0:
            return result

import multiprocessing as mp

def run_game_wrapper(game_id, result_dict):
    result = play_single_game(game_id)
    result_dict[game_id] = result

if __name__ == '__main__':
    num_games = 5
    manager = mp.Manager()
    results = manager.dict()
    processes = []

    start = time.time()

    for i in range(num_games):
        p = mp.Process(target=run_game_wrapper, args=(i, results))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print("Wins for Player 1:", list(results.values()).count(1))
    print("Wins for Player -1:", list(results.values()).count(-1))
    print("Draws:", list(results.values()).count(1e-4))
    print("Total time taken:", time.time() - start)
