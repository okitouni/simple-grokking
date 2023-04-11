# %%
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from data import prepare_nuclear_data, train_test_split
from sklearn.preprocessing import MinMaxScaler
from argparse import Namespace
from log import Logger
import shutil, tqdm

# --------------------------------------------- DATA
args = Namespace(
    TARGETS_CLASSIFICATION=
    # {"stability": 1, "parity": 1, "spin": 1, "isospin": 1},
    {},
    TARGETS_REGRESSION=
    # {
    #     "z": 1,
    #     "n": 1,
    #     "binding_energy": 1,
    #     "radius": 1,
    # },
    {
        "z": 1,
        "n": 1,
        "binding_energy": 1,
        "radius": 1,
        "half_life_sec": 1,
        "abundance": 1,
        "qa": 1,
        "qbm": 1,
        "qbm_n": 1,
        "qec": 1,
        "sn": 1,
        "sp": 1,
    },
    DEV="cpu",
)

data = prepare_nuclear_data(args, scaler=MinMaxScaler())
tasks = list(args.TARGETS_REGRESSION.keys())
num_tasks = len(args.TARGETS_REGRESSION)
P = data.X.max() + 1
X = torch.vstack(
    [torch.cat((x, torch.tensor([i]))) for x in data.X for i in range(num_tasks)]
)
X[:, -1] += P
y = data.y.flatten()
na_mask = torch.isnan(y)
X = X[~na_mask]
y = y[~na_mask].view(-1, 1)
train_frac = 0.8
shuffle = torch.randperm(len(X))
X_train, y_train = (
    X[shuffle[: int(len(X) * train_frac)]],
    y[shuffle[: int(len(X) * train_frac)]],
)
X_val, y_val = (
    X[shuffle[int(len(X) * train_frac) :]],
    y[shuffle[int(len(X) * train_frac) :]],
)
# inverse_transform = data.feature_transformer.inverse_transform


# %%
# --------------------------------------------- MODEL
class Model(nn.Module):
    def __init__(self, num_tasks, hidden_dim=64, num_layers=2):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(P + num_tasks, hidden_dim)
        self.nonlinear = torch.nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            # *[SimpleBlock(hidden_dim, hidden_dim) for _ in range(num_layers)]
            *[ResidualBlock(hidden_dim) for _ in range(num_layers)],
        )
        self.readout = nn.Linear(hidden_dim, num_tasks)

    def forward(self, x):
        task = x[:, -1] - P
        x = self.embedding(x).flatten(start_dim=1)
        x = self.nonlinear(x)
        x = self.readout(x)
        x = x[torch.arange(len(x)), task]
        return x.view(-1, 1)


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


# %% RUN
if __name__ == "__main__":
    torch.manual_seed(2)
    log = True
    name = f"nuclear"
    metrics = ["epoch", "train_loss", "val_loss"]
    metrics.extend([f"{target}_train_loss" for target in args.TARGETS_REGRESSION])
    metrics.extend([f"{target}_val_loss" for target in args.TARGETS_REGRESSION])
    logger = Logger(name, metrics) if log else None
    if log:
        shutil.copy(__file__, logger.root)

    # Hyperparameters
    num_epochs = 3000
    learning_rate = 1e-4
    weight_decay = 1e-1

    # Data
    loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=4, shuffle=True
    )
    model = Model(num_tasks=num_tasks)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, num_epochs * len(loader)
    )
    criterion = nn.MSELoss()

    def loss_by_function(y_pred, y, x):
        losses = [-1] * num_tasks
        for i in range(num_tasks):
            y_pred_ = y_pred[x[:, 2] == i + P]
            y_ = y[x[:, 2] == i + P]
            losses[i] = criterion(y_pred_, y_)
        return losses

    # Train

    title = "Train Loss | Val Loss"
    title += " | Subtasks:  " + " | ".join(tasks)
    print(title)
    pbar_train = tqdm.trange(num_epochs, leave=True, position=1, bar_format="{l_bar}")
    pbar_val = tqdm.trange(num_epochs, leave=True, position=0, bar_format="{l_bar}")
    pbar = tqdm.trange(num_epochs, leave=True, position=3, bar_format="{l_bar}{bar:20}{r_bar}")
    for epoch in pbar:
        for x, y in loader:
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
        with torch.no_grad():
            y_pred = model(X_train)
            loss = criterion(y_pred, y_train)
            train_losses = loss_by_function(y_pred, y_train, X_train)

            y_pred = model(X_val)
            val_loss = criterion(y_pred, y_val)
            val_losses = loss_by_function(y_pred, y_val, X_val)

        msg = f"{loss:10.2e} | {val_loss:>8.2e}"
        pbar.set_description_str(msg)
        pbar_train.set_description_str("TRAIN " + " | ".join([f"{l:8.2e}" for l in train_losses]))
        pbar_val.set_description_str("VAL " + " | ".join([f"{l:8.2e}" for l in val_losses]))

        if log:
            logger.log(
                model=None,
                epoch=epoch,
                train_loss=loss,
                val_loss=val_loss,
                **{
                    f"{target}_train_loss": train_losses[i]
                    for i, target in enumerate(tasks)
                },
                **{
                    f"{target}_val_loss": val_losses[i]
                    for i, target in enumerate(tasks)
                },
            )
# %%
if log:
    import os
    torch.save(model.state_dict(), os.path.join(logger.root, "model.pt"))