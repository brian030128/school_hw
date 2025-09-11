from gobang.game import GobangGame
from gobang.players import HumanGobangPlayer, RandomPlayer, AlphaZeroPlayer
from gobang.min_max_player import MinMaxPlayer
from net import NeuralNet
import torch

game = GobangGame()
human = HumanGobangPlayer(game)
greedy = RandomPlayer(game)
minmax = MinMaxPlayer(game, search_depth=2)

device = "cuda" if torch.cuda.is_available() else "cpu"
nn = NeuralNet(game).to(device)
nn.load_state_dict(torch.load("best.pth", map_location=torch.device(device)))
alphago = AlphaZeroPlayer(game, nn)

while True:
    board = game.getInitBoard()
    player = -1
    while True:
        if player == 1:
            action = human.play(board, player)
        else:
            action = alphago.play(board, player, temp=0)
        board, player = game.getNextState(board, player, action)
        game.display(board)
        result = game.getGameEnded(board, player)
        if result != 0:
            print("Result", result)
            break
    print("Game End")





