import time
import torch
from torch import nn
from torch.utils.data import Dataset

class ChessDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ChessCNN(nn.Module):
    def __init__(self, num_classes):
        super(ChessCNN, self).__init__()
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(13, 64, 3, 1, 1)   # input channels = 13 (bitboard channels)
        self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=50, device="cpu"):
    model.to(device)

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        scheduler.step()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100 * correct / total
        epoch_time = time.time() - start_time

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Loss: {epoch_loss:.4f} "
              f"Acc: {epoch_acc:.2f}% "
              f"Time: {epoch_time:.2f}s")

