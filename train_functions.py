# %%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import tqdm
from log import Logger
import os, shutil

functions_ = [
    lambda x, y: x + y,
    lambda x, y: abs(x - y),
    lambda x, y: (x + y) ** (2 / 3),
    lambda x, y: torch.log(x + y + 1),
    lambda x, y: torch.exp(-(x + y)/2),
]

metrics = ["epoch", "train_loss", "val_loss"]
metrics.extend([f"{i}_train_loss" for i in range(len(functions_))])
metrics.extend([f"{i}_val_loss" for i in range(len(functions_))])

def run(use_functions, functions=functions_):
    functions = [f for f, use in zip(functions, use_functions) if use]

    torch.manual_seed(2)
    log = True
    name = f"functions_{''.join([str(int(use)) for use in use_functions])}"
    logger = Logger(name, metrics) if log else None
    if log:
        shutil.copy(__file__, logger.root)
    # Modules
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

    SimpleBlock = lambda i, o: nn.Sequential(nn.Linear(i, o), nn.ReLU())

    # Hyperparameters
    num_epochs = 1000
    learning_rate = 1e-3
    weight_decay = 1e-5

    # Data
    P = 100
    train_frac = 0.01

    X = torch.cartesian_prod(
        torch.arange(P), torch.arange(P), torch.arange(len(functions))
    )
    y = torch.vstack([functions[idx](x, y) for x, y, idx in X]).float()
    for i in range(len(functions)):
        y[i :: len(functions)] = (
            y[i :: len(functions)] - y[i :: len(functions)].min()
        ) / (y[i :: len(functions)].max() - y[i :: len(functions)].min())
    X[:, 2] += P

    shuffle = torch.randperm(len(X))
    X, y = X[shuffle], y[shuffle]
    X_train, X_val = X[: int(train_frac * len(X))], X[int(train_frac * len(X)) :]
    y_train, y_val = y[: int(train_frac * len(y))], y[int(train_frac * len(y)) :]

    # Model
    class Model(nn.Module):
        def __init__(self, hidden_dim=256, num_layers=1):
            super(Model, self).__init__()
            self.embedding = nn.Embedding(P + len(functions), hidden_dim)
            self.nonlinear = torch.nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                *[SimpleBlock(hidden_dim, hidden_dim) for _ in range(num_layers)]
                # *[ResidualBlock(hidden_dim) for _ in range(num_layers)],
            )
            self.readout = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            x = self.embedding(x).flatten(start_dim=1)
            x = self.nonlinear(x)
            x = self.readout(x)
            return x

    model = Model()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = nn.MSELoss()
    def loss_by_function(y_pred, y):
        losses = [-1] * len(functions_)
        write_idx = [ i for i, use in enumerate(use_functions) if use]
        for i, write_idx in enumerate(write_idx):
            losses[write_idx] = criterion(y_pred[i::len(functions)], y[i::len(functions)])
        return losses
    # Train

    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=4, shuffle=True)
    if __name__ == "__main__":
        title = "Train Loss | Val Loss"
        title += " | " + " | ".join([f"{i}_train_loss" for i in range(len(functions_)) if use_functions[i]])
        title += " | " + " | ".join([f"{i}_val_loss" for i in range(len(functions_)) if use_functions[i]])
        print(title)
        pbar = tqdm.trange(num_epochs, leave=True, position=0)
        for epoch in pbar:
            for x, y in loader:
                optimizer.zero_grad()
                y_pred = model(x)
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                y_pred = model(X_train)
                loss = criterion(y_pred, y_train)
                train_losses = loss_by_function(y_pred, y_train)
            # optimizer.zero_grad()
            # y_pred = model(X_train)
            # loss = criterion(y_pred, y_train)
            # loss.backward()
            # optimizer.step()
            # train_losses = loss_by_function(y_pred, y_train)

            with torch.no_grad():
                y_pred = model(X_val)
                val_loss = criterion(y_pred, y_val)
                val_losses = loss_by_function(y_pred, y_val)

            msg = f"{loss:10.2e} | {val_loss:>8.2e}"
            msg += " | " + " | ".join([f"{l:8.2e}" for l in train_losses + val_losses])
            pbar.set_description(msg)

            if log:
                logger.log(
                    model=None,
                    epoch=epoch,
                    train_loss=loss,
                    val_loss=val_loss,
                    **{
                        f"{i}_train_loss": train_losses[i]
                        for i in range(len(train_losses))
                    },
                    **{f"{i}_val_loss": val_losses[i] for i in range(len(val_losses))},
                )


# %%
if __name__ == "__main__":
    list_uses = [[False] * len(functions_) for _ in  range(len(functions_))]
    for i in range(len(functions_)):
        list_uses[i][i] = True
    list_uses += [[True] * len(functions_)]
for use_functions in list_uses:
    run(use_functions, functions_)
# %%
# Plot accuracies
# read data
import pandas as pd
from matplotlib import pyplot as plt
log_dirs = ["log/functions_" + "".join([str(int(use)) for use in use_functions]) for use_functions in list_uses]
fig, ax = plt.subplots()
for log in log_dirs[:-1]:
    metrics = pd.read_csv(log + "/metrics.csv", index_col=False)
    plt.plot(metrics["epoch"], metrics["val_loss"], ls="--", label="trained on " + log.split("_")[-1], c=f"C{log_dirs.index(log)}")
log = log_dirs[-1]
metrics = pd.read_csv(log + "/metrics.csv", index_col=False)
for i in range(len(functions_)):
    plt.plot(metrics["epoch"], metrics[f"{i}_val_loss"], ls="-")
# add solid line legend
plt.plot([], [], ls="-", c="k", label=f"Trained on all")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xscale("log")
plt.yscale("log")
plt.title("Validation Loss by Function")
plt.savefig("functions_loss.pdf")

# %%
