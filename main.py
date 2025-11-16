
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from ChessCNN import ChessDataset, ChessCNN, train_model, ChessDataset
from Helpers import create_fen, all_moves, fen_to_tensor, move_to_index
pd.set_option('display.max_colwidth', None)  # Don't truncate column contents

DRY_RUN = True  # Set to False for full training

moves = all_moves() #4033

# Load CSV as before
df = pd.read_csv("chess_dataset.csv")

all_moves_list = all_moves()

# Limit the dataframe and moves for testing
if DRY_RUN:
    df = df.sample(n=5000, random_state=42)

X = []
y = []

for _, row in df.iterrows():
    fen_tensor = torch.tensor(fen_to_tensor(row['fen']), dtype=torch.float32)
    label = torch.tensor(move_to_index(row['predicted_move'], all_moves_list), dtype=torch.int64)
    X.append(fen_tensor)
    y.append(label)

# Optionally convert to torch tensors
X = torch.stack(X)   # shape: (num_samples, 13, 8, 8)
y = torch.tensor(y)  # shape: (num_samples,)


dataset = ChessDataset(X, y)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

num_classes = 4033  # Number of possible moves (for example)
device = "cuda" if torch.cuda.is_available() else "cpu"

model = ChessCNN(num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=20, device=device)

torch.save(model.state_dict(), "chess_cnn.pth")




