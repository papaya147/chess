import chess
import numpy as np
import torch

device = 'mps'

def board_to_tensor(board: chess.Board):
    tensor = np.zeros((12, 8, 8), dtype=np.float32)

    piece_map = board.piece_map()

    for square, piece in piece_map.items():
        row = square // 8
        col = square % 8

        offset = 0 if piece.color == chess.WHITE else 6
        if piece.piece_type == chess.PAWN:
            tensor[offset, row, col] = 1
        elif piece.piece_type == chess.BISHOP:
            tensor[offset + 1, row, col] = 1
        elif piece.piece_type == chess.KNIGHT:
            tensor[offset + 2, row, col] = 1
        elif piece.piece_type == chess.ROOK:
            tensor[offset + 3, row, col] = 1
        elif piece.piece_type == chess.QUEEN:
            tensor[offset + 4, row, col] = 1
        elif piece.piece_type == chess.KING:
            tensor[offset + 5, row, col] = 1

    return torch.tensor(tensor, dtype=torch.float32, device=device)

def move_index_mappings():
    move_to_index, index_to_move = {}, {}

    squares = list(chess.SQUARES)

    index = 0

    for from_square in squares: # regular from -> to moves
        for to_square in squares:
            if from_square == to_square:
                continue
            move = f'{chess.square_name(from_square)}{chess.square_name(to_square)}'
            move_to_index[move] = index
            index_to_move[index] = move
            index += 1

    squares = range(48, 55 + 1) # promotions for white, has to be on rank 7, squares a7 to h7 -> 48 to 55
    for from_square in squares[1:-1]: # b7 to g7 have 3 promo files: left, straight, right
        for d_square in range(7, 9 + 1):
            for promo in ['b', 'n', 'r', 'q']:
                move = f'{chess.square_name(from_square)}{chess.square_name(from_square + d_square)}{promo}'
                move_to_index[move] = index
                index_to_move[index] = move
                index += 1
    from_square = squares[0]
    for d_square in range(8, 9 + 1): # a7 has 2 promo files: straight, right
        for promo in ['b', 'n', 'r', 'q']:
            move = f'{chess.square_name(from_square)}{chess.square_name(from_square + d_square)}{promo}'
            move_to_index[move] = index
            index_to_move[index] = move
            index += 1
    from_square = squares[-1]
    for d_square in range(7, 8 + 1): # h7 has 2 promo files: left, straight
        for promo in ['b', 'n', 'r', 'q']:
            move = f'{chess.square_name(from_square)}{chess.square_name(from_square + d_square)}{promo}'
            move_to_index[move] = index
            index_to_move[index] = move
            index += 1

    squares = range(8, 15 + 1) # promotions for black, has to be on rank 2, squares a2 to h2 -> 8 to 15
    for from_square in squares[1:-1]: # b2 to g2 have 3 promo files: left, straight, right
        for d_square in range(-9, -7 + 1):
            for promo in ['b', 'n', 'r', 'q']:
                move = f'{chess.square_name(from_square)}{chess.square_name(from_square + d_square)}{promo}'
                move_to_index[move] = index
                index_to_move[index] = move
                index += 1
    from_square = squares[0]
    for d_square in range(-8, -7 + 1): # a2 has 2 promo files: straight, right
        for promo in ['b', 'n', 'r', 'q']:
            move = f'{chess.square_name(from_square)}{chess.square_name(from_square + d_square)}{promo}'
            move_to_index[move] = index
            index_to_move[index] = move
            index += 1
    from_square = squares[-1]
    for d_square in range(-9, -8 + 1): # h2 has 2 promo files: left, straight
        for promo in ['b', 'n', 'r', 'q']:
            move = f'{chess.square_name(from_square)}{chess.square_name(from_square + d_square)}{promo}'
            move_to_index[move] = index
            index_to_move[index] = move
            index += 1

    return move_to_index, index_to_move

move_to_index, index_to_move = move_index_mappings()

def move_mask(board, n_moves=len(move_to_index)):
    mask = np.zeros(n_moves)
    for move in board.legal_moves:
        uci = move.uci()
        if uci in move_to_index:
            idx = move_to_index[uci]
            mask[idx] = 1
        else:
            print(f'{uci} not handled in moves')
    return mask

