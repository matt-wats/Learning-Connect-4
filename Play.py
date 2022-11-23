import torch

import Search
import Board
import Evaluator



def play_game(evaluator):

    board = Board.Board()

    positions = []

    while not board.get_is_terminal():

        root = Search.MCTS(board, evaluator, 8)
        move, state, _ = Search.choose_move(root)

        board = state
        positions.append(board.get_board().float())

    return positions, board.get_terminal_value()
