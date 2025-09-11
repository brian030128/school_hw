from gobang.game import GobangGame
from gobang.min_max_player import MinMaxPlayer
from gobang.players import HumanGobangPlayer
import numpy as np

game = GobangGame()
human = HumanGobangPlayer(game)
minmax = MinMaxPlayer(game, search_depth=2, random_move_prob=0.1)


while True:
    board = game.getInitBoard()
    player = 1
    game.display(board)

    while True:
        if player == -1:
            print("MinMax turn")
            action = minmax.play(board, player)
            y, x = divmod(action, game.n)
            print(f"MinMax plays: {y} {x}")
        else:
            action = human.play(board, player)

        board, player = game.getNextState(board, player, action)
        game.display(board)

        result = game.getGameEnded(board, player)
        if result != 0:
            print("Result", result)
            break

    again = input("Do you want to play again? (y/n): ").strip().lower()
    if again != 'y':
        print("Thanks for playing!")
        break
