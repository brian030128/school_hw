import numpy as np
from random import randint
import time
import copy
import random

level_settings = [
    (0.5, 0),
    (0.4, 0),
    (0.3, 0),
    (0.2, 0),
    (0, 0),
    (0, 1),
    (0, 2)
]

class LeveledMinMaxPlayer:
    def __init__(self, game, level: int=0):
        #level is 1~10
        self.inner = MinMaxPlayer(game, level_settings[level][1], level_settings[level][0])
    
    def play(self, board, player):
        return self.inner.play(board, player)


class MinMaxPlayer:
    def __init__(self, game, search_depth=2, random_move_prob=0.1):
        self.game = game
        self.search_depth = search_depth
        self.board_size = game.n
        #self.count = -1
        self.last_move = '--'
        self.random_move_prob = random_move_prob # 傳入 random_move_prob
        # Create alphabet mapping for board coordinates
        self.alphabet = [chr(ord('a') + i) for i in range(26)]
        self.alphabet_dict = {chr(ord('a') + i): i for i in range(26)}
    
    def play(self, board, player):
        """Make a move using minimax algorithm"""
        
        # 計算目前總共下了多少步（非0的格子）
        current_count = np.count_nonzero(board)

        # 第一手（這個 AI 是該局的第一個下棋者）
        if current_count == 0:
            x = int(self.board_size / 2)
            y = int(self.board_size / 2)
            move_coord = (y, x)
            action = self.board_size * y + x
            self.last_move = self.transform_to_board_format(x, y)
            return action

        move_coord = self.get_best_move(board, player)
        y, x = move_coord
        action = self.board_size * y + x
        self.last_move = self.transform_to_board_format(x, y)
        return action
    
    def transform_to_board_format(self, x, y):
        """Convert numeric coordinates to board format (e.g., 'e5')"""
        col = self.alphabet[x]
        row = y + 1
        return col + str(row)
    
    def get_best_move(self, board, player):
        """Find the best move using minimax with alpha-beta pruning"""
        # random_move_chance 的機率下會隨機下棋
        if random.random() <  self.random_move_prob:
            valid_moves = [(i, j) for i in range(self.board_size) for j in range(self.board_size) if board[i][j] == 0]
            if valid_moves:
                return random.choice(valid_moves)
        
        # 其他和原來相同
        # Determine search depth based on game progress
        current_count = np.count_nonzero(board)
        depth = 1
        if current_count < int(self.board_size * 3):
            depth = 1
        elif current_count < int(self.board_size * 4):
            depth = 2
        else:
            depth = self.search_depth

        best_val = float('-inf')
        best_move = None
        alpha = float('-inf')
        beta = float('inf')
        
        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i][j] == 0:
                    # Try this move
                    board[i][j] = player
                    # Get value from minimax
                    move_val = self.minimax(board, depth, False, alpha, beta, -player)
                    # Undo the move
                    board[i][j] = 0
                    
                    if move_val > best_val:
                        best_move = (i, j)
                        best_val = move_val
                    alpha = max(alpha, best_val)
        
        # If no best move found (should not happen), pick a random valid move
        if best_move is None:
            for i in range(self.board_size):
                for j in range(self.board_size):
                    if board[i][j] == 0:
                        return (i, j)
        
        return best_move
    
    def minimax(self, board, depth, is_maximizing, alpha, beta, player):
        """Minimax algorithm with alpha-beta pruning"""
        # Base case - evaluate board
        if depth == 0:
            return self.evaluate_board(board, player)
        
        if is_maximizing:
            value = float('-inf')
            for i in range(self.board_size):
                for j in range(self.board_size):
                    if board[i][j] == 0:
                        board[i][j] = player
                        value = max(value, self.minimax(board, depth - 1, False, alpha, beta, -player))
                        board[i][j] = 0
                        alpha = max(alpha, value)
                        if beta <= alpha:
                            break
            return value
        else:
            value = float('inf')
            for i in range(self.board_size):
                for j in range(self.board_size):
                    if board[i][j] == 0:
                        board[i][j] = player
                        value = min(value, self.minimax(board, depth - 1, True, alpha, beta, -player))
                        board[i][j] = 0
                        beta = min(beta, value)
                        if beta <= alpha:
                            break
            return value
    
    def evaluate_board(self, board, player):
        """Evaluate the board and return a score"""
        return self.sb(board, self.board_size, player)
    
    def sb(self, board, l, player):
        """Board evaluation function (kept from original implementation)"""
        mystring = ''
    
        def translate(cell):
            if cell == player:
                return '1'
            elif cell == -player:
                return '2'
            else:
                return str(int(cell))

        # rows
        for i in board:
            mystring += 'W'
            for x in i:
                mystring += translate(x)
        # columns
        for i in range(0, l):
            mystring += 'W'
            for j in range(0, l):
                mystring += translate(board[j][i])
        
        # diagonals
        diags = [board[::-1,:].diagonal(i) for i in range(-1*(l-1),l)]
        diags.extend(board.diagonal(i) for i in range(l-1,l*-1,-1))
        x = [n.tolist() for n in diags]
        for z in x:
            mystring += 'W'
            for y in z:
                mystring += translate(y)

        # Scoring patterns (1 is maximizing player, 2 is minimizing player)
        total_me = mystring.count('00011')*100 + \
            mystring.count('00100')*10 + \
            mystring.count('11000')*100 + \
            mystring.count('010010')*200 + \
            mystring.count('01010')*250 + \
            mystring.count('00011000')*1000 + \
            mystring.count('10101')*500 + \
            mystring.count('11010')*600 + \
            mystring.count('10110')*600 + \
            mystring.count('11100')*500 + \
            mystring.count('00111')*500 + \
            mystring.count('01110')*3000 + \
            mystring.count('011010')*900 + \
            mystring.count('010110')*900 + \
            mystring.count('11011')*2000 + \
            mystring.count('10111')*3500 + \
            mystring.count('11101')*3500 + \
            mystring.count('11110')*6000 + \
            mystring.count('01111')*6000 + \
            mystring.count('011110')*100000 + \
            mystring.count('11111')*10000000
            
        total_you = mystring.count('00022')*100 + \
            mystring.count('00200')*10 + \
            mystring.count('22000')*100 + \
            mystring.count('020020')*200 + \
            mystring.count('02020')*250 + \
            mystring.count('00022000')*1000 + \
            mystring.count('20202')*500 + \
            mystring.count('22020')*600 + \
            mystring.count('20220')*600 + \
            mystring.count('22200')*500 + \
            mystring.count('00222')*500 + \
            mystring.count('02220')*3000 + \
            mystring.count('022020')*900 + \
            mystring.count('020220')*900 + \
            mystring.count('22022')*2000 + \
            mystring.count('20222')*3500 + \
            mystring.count('22202')*3500 + \
            mystring.count('22220')*6000 + \
            mystring.count('02222')*6000 + \
            mystring.count('022220')*100000  + \
            mystring.count('22222')*10000000

        total = total_me - total_you
        return total