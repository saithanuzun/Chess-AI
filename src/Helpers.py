import csv
from typing import Any
import chess
import numpy as np
import pandas as pd

def create_fen(moves_string:str)-> tuple[list[Any], list[str]]:
    board = chess.Board()

    moves_list = moves_string.split()
    fen = []
    moves_uci = []
    fen.append(board.fen()) #first fen of board

    for move in moves_list:
        moves_uci.append(board.parse_san(move).uci())
        board.push_san(move)
        fen.append(str(board.fen()))

    moves_uci.append('ff')
    return fen, moves_uci

def all_moves() -> list[str]:
    files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    ranks = [str(i+1) for i in range(8)]

    board = [f + r for f in files for r in ranks]

    all_moves = [a + b for a in board for b in board if a!=b]

    all_moves.append('ff')

    return all_moves

def fen_to_tensor(fen: str) -> np.ndarray:
    board = chess.Board(fen)
    turn = board.turn
    tensor = np.zeros((13, 8, 8), dtype=np.float32)

    piece_to_index = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5
    }

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row = 7 - chess.square_rank(square)  # row 0 = top of board
            col = chess.square_file(square)
            idx = piece_to_index[piece.piece_type]
            if piece.color == chess.BLACK:
                idx += 6  # black pieces go to channels 6–11
            tensor[idx, row, col] = 1.0

        if turn == chess.WHITE:
            tensor[12, :, :] = 1.0  # all ones for White to move
        else:
            tensor[12, :, :] = 0.0  # all zeros for Black to move

    return tensor

def extract_fens_to_csv( path: str):
    games = pd.read_csv(path)['moves']

    data_flat = []

    for game_idx, game in games.items():
        fens, moves = create_fen(game)
        print(f"Processing game {game_idx + 1}/{len(games)}")  # which game
        for fen, move in zip(fens, moves):
            data_flat.append((fen, move))

    with open("chess_dataset.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fen", "predicted_move"])  # 2-column header
        writer.writerows(data_flat)

def move_to_index(move: str, moves: list[str]) -> int:
    move = move[:4]
    return moves.index(move)

