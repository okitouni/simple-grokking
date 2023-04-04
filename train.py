import torch
import torch.nn as nn
import tqdm
from log import Logger
from sys import argv

torch.manual_seed(2)
grok = "--grok" in argv
log = "--log" in argv
logger = Logger("grokking" if grok else "comprehension") if log else None

# Hyperparameters
num_epochs = 4000
learning_rate = 3e-2
weight_decay = 3e-2 if grok else 3

# Data - sum of two numbers mod 53
P = 53
train_frac = 0.8
X = torch.cartesian_prod(torch.arange(P), torch.arange(P))
y = (X[:, 0] + X[:, 1]) % P
shuffle = torch.randperm(len(X))
X, y = X[shuffle], y[shuffle]
X_train, X_val = X[: int(train_frac * len(X))], X[int(train_frac * len(X)) :]
y_train, y_val = y[: int(train_frac * len(y))], y[int(train_frac * len(y)) :]

# Model
class Model(nn.Module):
    def __init__(self, hidden_dim=256):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(P, hidden_dim)
        self.fc1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.readout = nn.Linear(hidden_dim, P)

    def forward(self, x):
        x = self.embedding(x).flatten(start_dim=1)
        x = torch.relu(self.fc1(x))
        x = self.readout(x)
        return x


model = Model(hidden_dim=128)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=learning_rate, weight_decay=weight_decay
)
criterion = nn.CrossEntropyLoss()

# Train
if __name__ == "__main__":
    print("Train Loss, Acc | Val Loss, Acc")
    pbar = tqdm.trange(num_epochs, leave=True, position=0)
    for epoch in pbar:
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
