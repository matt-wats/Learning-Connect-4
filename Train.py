import torch
import torch.nn as nn


import Play
import Board
import Search
import Evaluator


def play_game(evaluator):

    board = Board.Board()

    positions = []

    while not board.get_is_terminal():

        root = Search.MCTS(board, evaluator, 16)
        move, state, _ = Search.choose_move(root)

        board = state
        positions.append(board.get_board())

    return positions, board.get_terminal_value()



def get_game_data(evaluator):

    positions, score = Play.play_game(evaluator)

    x = Board.stack_positions(positions)
    y = torch.where(torch.arange(x.size(0)-1, -1, -1) % 2 == 0, score, 1-score).view(-1,1)

    return x,y




def train_game(evaluator, optimizer: torch.optim.AdamW):

    x, y = get_game_data(evaluator)

    optimizer.zero_grad()

    preds = evaluator(x)

    loss = nn.MSELoss()(preds,y)

    loss.backward()
    optimizer.step()

    return loss.item()


def train(evaluator, epochs: int, optimizer: torch.optim.AdamW):

    losses = []

    for i in range(epochs):
        print(f"Game #{i+1} / {epochs} |", end='\t')
        loss = train_game(evaluator, optimizer)
        losses.append(loss)

    return losses
