from gobang.game import GobangGame
from gobang.players import HumanGobangPlayer, RandomPlayer, AlphaZeroPlayer
from net import NeuralNet
import torch
import time

game = GobangGame()

device = "cuda" if torch.cuda.is_available() else "cpu"
nn1 = NeuralNet(game).to(device)
nn1.eval()
alphago1 = AlphaZeroPlayer(game, nn1)

nn2 = NeuralNet(game).to(device)
nn2.eval()
alphago2 = AlphaZeroPlayer(game, nn2)

start = time.time()


board = game.getInitBoard()
player = 1
while True:
    if player == 1:
        action = alphago1.play(board)
    else:
        action = alphago2.play(board)
    board, player = game.getNextState(board, player, action)
    result = game.getGameEnded(board, player)
    if result != 0:
        print("Result", result)
        break

print("Game End")
print("Total time taken:", time.time() - start)

