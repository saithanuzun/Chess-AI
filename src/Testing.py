import torch
from ChessCNN import ChessCNN
from Helpers import all_moves, fen_to_tensor

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the move list
all_moves_list = all_moves()  # same as during training

# Initialize the model
num_classes = len(all_moves_list)
model = ChessCNN(num_classes=num_classes).to(device)

# Load trained weights
model.load_state_dict(torch.load("../chess_cnn.pth", map_location=device))
model.eval()  # important: set to evaluation mode

test_fens = [
    # Starting position
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",

    # After 1.e4 e6 (French Defense)
    "rnbqkbnr/ppp1pppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",

    # After 1.e4 e5 2.Nf3 Nc6 3.Bb5 (Ruy Lopez)
    "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",

    # Midgame random position
    "r2q1rk1/ppp2ppp/2n2n2/2b1p3/4P3/2N1BN2/PPP2PPP/R2Q1RK1 w - - 6 10"
]

test_fens += [
    # Sicilian Defense, after 1.e4 c5
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",

    # After 1.d4 d5 2.c4 e6 (Queen's Gambit Declined)
    "rnbqkbnr/ppp1pppp/4p3/3p4/2P5/8/PP1P1PPP/RNBQKBNR w KQkq - 0 3",

    # King's Indian Defense, after 1.d4 Nf6 2.c4 g6
    "rnbqkb1r/pppp1ppp/5n2/8/2P5/8/PP1PPPPP/RNBQKBNR w KQkq - 2 3",

    # Midgame, attacking position
    "r1bq1rk1/ppp2ppp/2n1pn2/3p4/3P4/2P1PN2/P1Q2PPP/R1B1KB1R w KQ - 0 8",

    # Endgame, King + Pawn vs King
    "8/8/8/4k3/8/4P3/4K3/8 w - - 0 30",

    # Endgame, Rook + King vs King
    "8/8/8/4R3/4k3/8/4K3/8 w - - 0 50",

    # Complex middlegame, attacking kingside
    "r1bq1rk1/pp1n1ppp/2p1pn2/3p4/3P1B2/2N1PN2/PPQ2PPP/R3KB1R b KQ - 2 10",

    # Open position, early game
    "rnbqkbnr/pppp1ppp/8/4p3/3P4/8/PPP1PPPP/RNBQKBNR w KQkq e6 0 2"
]

test_fens += [
    # King + Pawn vs King (White to move, winning)
    "8/8/8/4k3/8/4P3/4K3/8 w - - 0 30",

    # King + Rook vs King (White to move, winning)
    "8/8/8/4R3/4k3/8/4K3/8 w - - 0 50",

    # King + Bishop vs King + Pawn (White to move, draw)
    "8/8/8/4k3/5p2/6B1/4K3/8 w - - 0 60",

    # King + Knight vs King + Pawn (White to move, tricky)
    "8/8/8/4k3/6p1/5N2/4K3/8 w - - 0 45",

    # Rook + Pawn vs Rook (Lucena position)
    "8/8/8/8/8/2R5/4k3/4K3 w - - 0 72",

    # Rook + Pawn vs Rook (Philidor position)
    "8/8/8/8/4k3/8/1P6/4K2R w - - 0 55",

    # King + 2 Pawns vs King (White to move, winning)
    "8/8/8/4k3/6P1/5P2/4K3/8 w - - 0 40",

    # King + Queen vs King + Pawn (White to move, winning)
    "8/8/8/4k3/5p2/8/4K3/3Q4 w - - 0 20",

    # Minor piece endgame: Bishop + Knight vs King
    "8/8/8/4k3/5N2/6B1/4K3/8 w - - 0 35",

    # Symmetrical King + Pawns endgame
    "8/8/8/4k3/3p4/3P4/4K3/8 w - - 0 25"
]



for fen in test_fens:
    fen_tensor = torch.tensor(fen_to_tensor(fen), dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(fen_tensor)
        predicted_index = torch.argmax(outputs, dim=1).item()
    print(f"FEN: {fen} -> Predicted move: {all_moves_list[predicted_index]}")

def ChessAI(fen):
    fen_tensor = torch.tensor(fen_to_tensor(fen), dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(fen_tensor)
        predicted_index = torch.argmax(outputs, dim=1).item()
        return all_moves_list[predicted_index]

