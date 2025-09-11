from gobang.board import Board 
from gobang.game import GobangGame

import torch
from torch import nn
import torch.nn.functional as F

class NeuralNet(nn.Module):

    def __init__(self, game: GobangGame, num_channels: int = 512, dropout: float = 0.3):
        super(NeuralNet, self).__init__()
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        self.conv1 = nn.Conv2d(1, num_channels, kernel_size=3, padding=1)  # same padding

        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)

        self.conv3 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=0)  # valid padding

        self.conv4 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=0)

        conv_output_size = num_channels * (self.board_x - 4) * (self.board_y - 4)

        self.fc1 = nn.Linear(conv_output_size, 1024)

        self.fc2 = nn.Linear(1024, 512)

        self.dropout = nn.Dropout(dropout)

        self.pi = nn.Linear(512, game.getActionSize())
        self.v = nn.Linear(512, 1)

    def forward(self, s):
        
        s = s.unsqueeze(1) # add channel dimension
        x = F.leaky_relu(self.conv1(s))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))

        x = x.view(x.size(0), -1)  # flatten

        x = self.dropout(F.leaky_relu(self.fc1(x)))
        x = self.dropout(F.leaky_relu(self.fc2(x)))

        pi = F.softmax(self.pi(x), dim=1)  # policy output
        v = torch.tanh(self.v(x))          # value output

        return pi, v            # batch_size x 1


    def predict(self, board: Board):
        """
        Returns:
            pi: a policy vector for the current board- a numpy array of length
                game.getActionSize
            v: a float in [-1,1] that gives the value of the current board
        """
                # game params
        board = torch.from_numpy(board).float().to(self.conv1.weight.device)
        board = board[torch.newaxis, :, :]
        pi, v = self(board)
        return pi[0].to("cpu").numpy(), v[0].to("cpu").numpy()