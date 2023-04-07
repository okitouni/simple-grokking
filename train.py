import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import tqdm
from log import Logger
from sys import argv
import os

torch.manual_seed(2)
log = "--log" in argv
name = os.environ["NAME"] if "NAME" in os.environ else "run"
logger = Logger(name) if log else None

# Hyperparameters
num_epochs = 50000
learning_rate = 2e-2
weight_decay = 0.5

# Data - sum of two numbers mod 53
Ps = torch.tensor([23, 29, 31, 37, 41, 43, 47, 53])
P = max(Ps)
train_frac = 0.3

X = torch.cartesian_prod(torch.arange(P), torch.arange(P), torch.arange(P, P + len(Ps)))
y = (X[:, 0] + X[:, 1]) 
print("original data size: ", len(y), end=" ")
keep = y >= max(Ps)
X, y = X[keep], y[keep]
y = y % Ps[X[:, 2] - P]
print("after filtering: ", len(y), "frac: ", f"{len(y) / len(keep):.2f}")
shuffle = torch.randperm(len(X))
X, y = X[shuffle], y[shuffle]
X_train, X_val = X[: int(train_frac * len(X))], X[int(train_frac * len(X)) :]
y_train, y_val = y[: int(train_frac * len(y))], y[int(train_frac * len(y)) :]

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, d_model):
        super(ResidualBlock, self).__init__()
        self.nonlinear = torch.nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.nonlinear(x) + x


# Model
class Model(nn.Module):
    def __init__(self, hidden_dim=256):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(P + len(Ps), hidden_dim)
        self.nonlinear = torch.nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            *[ResidualBlock(hidden_dim) for _ in range(0)],
        )
        self.readout = nn.Linear(hidden_dim, P)

    def forward(self, x):
        x = self.embedding(x).flatten(start_dim=1)
        x = self.nonlinear(x)
        x = self.readout(x)
        return x


model = Model(hidden_dim=64)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=learning_rate, weight_decay=weight_decay
)
criterion = nn.CrossEntropyLoss()

# Train

loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
if __name__ == "__main__":
    print("Train Loss, Acc | Val Loss, Acc")
    pbar = tqdm.trange(num_epochs, leave=True, position=0)
    for epoch in pbar:
        # for x, y in loader:
        #     optimizer.zero_grad()
        #     y_pred = model(x)
        #     loss = criterion(y_pred, y)
        #     loss.backward()
        #     optimizer.step()
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            train_acc = (y_pred.argmax(dim=1) == y_train).float().mean() * 100

            y_pred = model(X_val)
            val_loss = criterion(y_pred, y_val)
            val_acc = (y_pred.argmax(dim=1) == y_val).float().mean() * 100

        msg = f"{loss:10.2f}, {train_acc:>3.0f} | {val_loss:>8.2f}, {val_acc:>4.0f}"
        pbar.set_description(msg)

        if log:
            logger.log(
                model=model,
                epoch=epoch,
                train_loss=loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
            )
