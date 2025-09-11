

board = [['.' for _ in range(15)] for _ in range(15)]
num = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "H": 7,
    "I": 8,
    "J": 9,
    "K": 10,
    "L": 11,
    "M": 12,
    "N": 13,
    "O": 14,
}


def print_board():
    print()
    for i in range(15):
        print(board[i])
    print()

dirs = [(1, 0), (0, 1), (1, 1), (1, -1)]

def check_winner(x, y, player):
    for dir in dirs:
        tempX = x
        tempY = y
        linked = 1
        while True:
            tempX += dir[0]
            tempY += dir[1]
            if tempX >= 15 or tempY >= 15:
                break
            if board[tempX][tempY] == player:
                linked += 1
            else:
                break
        tempX = x
        tempY = y
        while True:
            tempX += -1 * dir[0]
            tempY += -1 * dir[1]
            if tempX >= 15 or tempY >= 15:
                break
            if board[tempX][tempY] == player:
                linked += 1
            else:
                break
        #print(linked)
        if linked >= 5:
            return 1 if player == 'B' else -1


    return 0

player = 'B'
step = 1

while True:
    line = input()
    if line.strip(' ') == "end":
        print("The game is tie.")
        break
    args = line.split(" ")
    x = num[args[0]]
    y = int(args[1])

    if x >= 15 or y >= 15 or board[x][y] != '.' :
        print(f"Invalid move at step {step}.")
        break
    step += 1

    board[x][y] = player
    result = check_winner(x, y, player)
    if result == -1:
        print("The winner is white.")
        break
    if result == 1:
        print("The winner is black.")
        break
    #print_board()

    player = 'W' if player == 'B' else 'B'
