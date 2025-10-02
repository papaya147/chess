import numpy as np
from abc import ABC, abstractmethod

def in_bounds(board, row, col):
    n = len(board)
    if row < 0:
        return False
    if col < 0:
        return False
    if row >= n:
        return False
    if col >= n:
        return False
    return True

def move_score(board, color, row, col):
    curr_p = board[row, col]
    if curr_p is None: # piece can move, score is 1 for moving
        return 1
    if curr_p.color == color: # piece can't move there
        return -1
    return 1 + curr_p.value # 1 point for moving the piece, points for cutting a piece of opposite color


def continuous_tiles_till_block(board, color, start_row, start_col, row_step, col_step):
    r = start_row + row_step
    c = start_col + col_step

    tiles = {}

    while in_bounds(board, r, c):
        score = move_score(board, color, r, c)
        if score != -1: # add tile score that is valid
            tiles[(r, c)] = score
        if score != 1: # can't move any further
            break

        r += row_step
        c += col_step

    # the last tile was in bounds
    return tiles

class Piece(ABC):
    def __init__(self, color, row, col, value, track_has_moved=False):
        self.color = color
        self.value = value
        self.row = row
        self.col = col
        if track_has_moved:
            self.has_moved = False

    @abstractmethod
    def possible_moves(self, board):
        """
        Return dictionary of {possible moves: score} for this piece.
        """
        pass

    def valid_move(self, board, row, col):
        pms = self.possible_moves(board)
        return pms.get((row, col)) is not None
    
    def move(self, board, row, col):
        pms = self.possible_moves(board)
        if pms.get((row, col)) is None: # move is invalid
            return board
        board[self.row, self.col] = None  # clear old position
        board[row, col] = self
        self.row = row  # update piece's position
        self.col = col
        if hasattr(self, 'has_moved'):
            self.has_moved = True
        return board


class Rook(Piece):
    def __init__(self, color, row, col):
        super().__init__(color, row, col, value=5, track_has_moved=True)

    def possible_moves(self, board):
        # up
        tiles = continuous_tiles_till_block(board, self.color, self.row, self.col, -1, 0)
        # right
        tiles |= continuous_tiles_till_block(board, self.color, self.row, self.col, 0, 1)
        # down
        tiles |= continuous_tiles_till_block(board, self.color, self.row, self.col, 1, 0)
        # left
        tiles |= continuous_tiles_till_block(board, self.color, self.row, self.col, 0, -1)
        return tiles

class Bishop(Piece):
    def __init__(self, color, row, col):
        super().__init__(color, row, col, value=3, track_has_moved=False)

    def possible_moves(self, board):
        # top right
        tiles = continuous_tiles_till_block(board, self.color, self.row, self.col, -1, 1)
        # bottom right
        tiles |= continuous_tiles_till_block(board, self.color, self.row, self.col, 1, 1)
        # bottom left
        tiles |= continuous_tiles_till_block(board, self.color, self.row, self.col, 1, -1)
        # top left
        tiles |= continuous_tiles_till_block(board, self.color, self.row, self.col, -1, -1)
        return tiles
    
class Queen(Piece):
    def __init__(self, color, row, col):
        super().__init__(color, row, col, value=9, track_has_moved=False)

    def possible_moves(self, board):
        # up
        tiles = continuous_tiles_till_block(board, self.color, self.row, self.col, -1, 0)
        # top right
        tiles |= continuous_tiles_till_block(board, self.color, self.row, self.col, -1, 1)
        # right
        tiles |= continuous_tiles_till_block(board, self.color, self.row, self.col, 0, 1)
        # bottom right
        tiles |= continuous_tiles_till_block(board, self.color, self.row, self.col, 1, 1)
        # down
        tiles |= continuous_tiles_till_block(board, self.color, self.row, self.col, 1, 0)
        # bottom left
        tiles |= continuous_tiles_till_block(board, self.color, self.row, self.col, 1, -1)
        # left
        tiles |= continuous_tiles_till_block(board, self.color, self.row, self.col, 0, -1)
        # top left
        tiles |= continuous_tiles_till_block(board, self.color, self.row, self.col, -1, -1)
        return tiles
    
class King(Piece):
    def __init__(self, color, row, col):
        super().__init__(color, row, col, value=50, track_has_moved=True)

    def dict_move_score(self, board, row, col):
        if in_bounds(board, row, col):
            score = move_score(board, self.color, row, col)
            if score != -1:
                return {(row, col): score}
        return {}

    def possible_moves(self, board):
        # up
        tiles = self.dict_move_score(board, self.row - 1, self.col)
        # top right
        tiles |= self.dict_move_score(board, self.row - 1, self.col + 1)
        # right
        tiles |= self.dict_move_score(board, self.row, self.col + 1)
        # bottom right
        tiles |= self.dict_move_score(board, self.row + 1, self.col + 1)
        # down
        tiles |= self.dict_move_score(board, self.row + 1, self.col)
        # bottom left
        tiles |= self.dict_move_score(board, self.row + 1, self.col - 1)
        # left
        tiles |= self.dict_move_score(board, self.row, self.col - 1)
        # top left
        tiles |= self.dict_move_score(board, self.row - 1, self.col - 1)
        # castle
        if not self.has_moved and not attacked(board, self.color, self.row, self.col): # possible if king not moved and not attacked currently
            rooks = not_moved_rooks(board, self.color)
            for r in rooks: # make sure tiles in between are not attacked, rooks have to be in same row, diff col
                valid_path = True # assume path is free
                from_col = min(r.col, self.col)
                to_col = max(r.col, self.col)
                if self.col == to_col:
                    from_col += 1

                for i in range(from_col, to_col):
                    if board[self.row, i] is not None or attacked(board, self.color, self.row, i):
                        valid_path = False
                
                if valid_path:
                    if self.col > r.col:
                        tiles[(self.row, self.col - 2)] = 2
                    else:
                        tiles[(self.row, self.col + 2)] = 2
        
        return tiles
    
    def move(self, board, row, col):
        pms = self.possible_moves(board)
        score = pms.get((row, col))
        if score is None: # move is invalid
            return board
        if score == 1:
            return super().move(board, row, col)
        # castling
        # calculate rooks new position
        n = len(board)
        if col < self.col: # king's moving left, select the left rook
            r = board[self.row, 0]
            rook_old_c = 0
            rook_new_c = col + 1
        else: # king's moving right, select the right rook
            r = board[self.row, n - 1]
            rook_old_c = n - 1
            rook_new_c = col - 1
        board[self.row, self.col] = None  # clear old king position
        board[row, col] = self
        self.row = row  # update king's position
        self.col = col
        self.has_moved = True
        board[self.row, rook_old_c] = None  # clear old rook position
        board[row, rook_new_c] = self
        r.row = row  # update rook's position
        r.col = rook_new_c
        r.has_moved = True
        return board


def not_moved_rooks(board, color):
    rooks = []
    n = len(board)
    for i in range(n):
        for j in range(n):
            p = board[i, j]
            if p is None:
                continue
            if p.color == color and p.value == 5 and not p.has_moved:
                rooks.append(p)
    return rooks

def attacked(board, color, row, col):
    for p in board_pieces(board, color):
        if p.valid_move(board, row, col):
            return True
    return False


def board_pieces(board, opp_color):
    pieces = []
    n = len(board)
    for i in range(n):
        for j in range(n):
            p = board[i, j]
            if p is None:
                continue
            if p.color != opp_color:
                pieces.append(p)
    return pieces

r1 = Rook('b', 0, 0)
k1 = King('b', 0, 3)
board = np.array([
    [r1, None, None, k1], 
    [None, None, None, None], 
    [None, None, None, None], 
    [None, None, None, None]])
print(r1.possible_moves(board))
print(k1.possible_moves(board))
board = k1.move(board, 0, 1)
print(r1.possible_moves(board))
print(k1.possible_moves(board))