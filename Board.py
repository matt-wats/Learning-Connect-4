import torch

# input is a list of positions (tensors) with length n, outputs a tensor with a new dim0 of size n
def stack_positions(positions):

    return torch.stack(positions)


class Board():
    
    def __init__(self, board=None) -> None:
        self.height = 6
        self.width = 7
        self.channels = 2

        if board == None:
            self.board = torch.zeros(size=(self.channels,self.height,self.width), dtype=int)
        else:
            self.board = board

    
    # evaluator is some function (NN Model) that assigns a value to a position
    # returns evaluation of position
    def evaluate(self, evaluator):
        board = self.get_board().float().view(1, self.channels, self.height, self.width)
        return evaluator(board)


    def get_board(self) -> torch.Tensor:
        return self.board.detach().clone()

    
    # returns list of possible actions that can be made for a given board
    def get_moves(self) -> list:

        # [2x6x7] -> [6x7] -> [7]
        # board -> is full bool -> first open pos in column
        is_taken = self.board.max(dim=0).values
        row_positions = self.height-1 - is_taken.flip((0)).argmin(dim=0)
        open_cols = (1-is_taken[0,:]).nonzero()

        # for each col ind and row val in col_open_pos, if available row, put in list 
        moves = [[row_positions[c].item(), c.item()] for c in open_cols]

        return moves

    # returns lists of possible moves from a given board and the actions that result from each move
    def get_states(self) -> list:

        moves = self.get_moves()

        states = [self.apply_move(move) for move in moves]

        return states, moves

    
    # returns a NEW board after applying a given move
    def apply_move(self, move) -> torch.Tensor:

        new_board = Board(self.get_board())

        r, c = move
        new_board.board[0,r,c] = 1

        new_board.flip_board()

        return new_board


    # flips player 1s pieces for player 2s
    def flip_board(self) -> None:
        self.board = self.board.flip((0))





    def get_is_terminal(self) -> bool:
        return self.check_tie() or self.check_win()

    # assumes game is in terminal state
    def get_terminal_value(self):
        return 0.5 if self.check_tie() else 1.


    def check_tie(self) -> bool:
        return self.board[:,0].sum() == self.width


    def check_win(self) -> bool:

        # vertical connect 4s
        for i in range(self.height):
            for j in range(self.width-3):
                if self.board[1, i,j:j+4].min() == 1:
                    return True

        
        # vertical connect 4s
        for i in range(self.height-3):
            for j in range(self.width):
                if self.board[1, i:i+4, j].min() == 1:
                    return True


        # diagonal connect 4s
        for i in range(self.height-3):
            for j in range(self.width-3):

                ar = torch.arange(0,4)

                if self.board[1, i+ar, j+ar].min() == 1:
                    return True

                if self.board[1, i+ar, j+3-ar].min() == 1:
                    return True

        return False


