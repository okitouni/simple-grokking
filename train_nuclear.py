# %%
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from data import prepare_nuclear_data, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from argparse import Namespace
from log import Logger
import shutil, tqdm
import wandb

# Hyperparameters
hps = Namespace(
    num_epochs=3000,
    learning_rate=1e-3,
    weight_decay=1e-1,
    train_frac=0.05,
    batch_size=4,
    hidden_dim=64,
    num_layers=2,
    TARGETS_CLASSIFICATION=
    # {"stability": 1, "parity": 1, "spin": 1, "isospin": 1},
    {},
    TARGETS_REGRESSION={
        "z": 1,
        "n": 1,
        "binding": 1,
        "radius": 1,
        "sum_zn": 1,
        # "diff_zn": 1,
        # "n_mod_2": 1,
        # "z_mod_2": 1,
        # "sum_mod_2": 1,
        #     "half_life_sec": 1,
        #     "abundance": 1,
        #     "qa": 1,
        #     "qbm": 1,
        #     "qbm_n": 1,
        #     "qec": 1,
        #     "sn": 1,
        #     "sp": 1,
    },
    DEV=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)
WANDB = True
if WANDB:
    run = wandb.init(
        project="ai-nuclear-simple",
        config=hps,
        group="new-age-task-embs",
        notes="test-run",
        tags=["test-run"]
        )

data = prepare_nuclear_data(hps, scaler=MinMaxScaler())
tasks = list(hps.TARGETS_REGRESSION.keys())
num_tasks = len(hps.TARGETS_REGRESSION)
Z, N = data.X.amax(0) + 1
P = Z + N
X = torch.vstack(
    [
        torch.tensor([x[0], x[1] + Z, i])
        for x in data.X
        for i in range(P, P + num_tasks)
    ]
)
y = data.y.flatten()
na_mask = torch.isnan(y)
X = X[~na_mask]
y = y[~na_mask].view(-1, 1)
shuffle = torch.randperm(len(X))
train_idx = int(len(X) * hps.train_frac)

X_train, y_train = (
    X[shuffle[:train_idx]],
    y[shuffle[:train_idx]],
)
X_val, y_val = (
    X[shuffle[train_idx : train_idx * 2]],
    y[shuffle[train_idx : train_idx * 2]],
)
X_train, y_train = X_train.to(hps.DEV), y_train.to(hps.DEV)
X_val, y_val = X_val.to(hps.DEV), y_val.to(hps.DEV)

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


def start_run():
    torch.manual_seed(2)
    log = True
    name = f"nuclear"
    metrics = ["epoch", "train_loss", "val_loss"]
    metrics.extend([f"{target}_train_loss" for target in hps.TARGETS_REGRESSION])
    metrics.extend([f"{target}_val_loss" for target in hps.TARGETS_REGRESSION])
    logger = Logger(name, metrics) if log else None
    if log:
        shutil.copy(__file__, logger.root)

    # Data
    loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=hps.batch_size, shuffle=True
    )
    model = Model(num_tasks=num_tasks)
    model.to(hps.DEV)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=hps.learning_rate, weight_decay=hps.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, hps.num_epochs * len(loader)
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
    title = "Sub " + " | ".join(f"{task:^8}" for task in tasks)
    print(title)
    pbar_train = tqdm.trange(
        hps.num_epochs, leave=True, position=1, bar_format="{l_bar}", dynamic_ncols=True
    )
    pbar_val = tqdm.trange(
        hps.num_epochs, leave=True, position=0, bar_format="{l_bar}", dynamic_ncols=True
    )
    pbar = tqdm.trange(
        hps.num_epochs,
        leave=True,
        position=3,
        bar_format="{l_bar}{bar:20}{r_bar}",
        dynamic_ncols=True,
    )
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

        msg = f"Trn Loss {loss:<8.2e} | Val Loss {val_loss:<8.2e}"
        pbar.set_description_str(msg)
        pbar_train.set_description_str(
            "Trn " + " | ".join([f"{l:<8.2e}" for l in train_losses])
        )
        pbar_val.set_description_str(
            "Val " + " | ".join([f"{l:<8.2e}" for l in val_losses])
        )

        if log:
            logger.log(
                model=None,
                **(
                    metrics_dict := {
                        "epoch": epoch,
                        "train_loss": loss,
                        "val_loss": val_loss,
                        **{
                            f"{target}_train_loss": train_losses[i]
                            for i, target in enumerate(tasks)
                        },
                        **{
                            f"{target}_val_loss": val_losses[i]
                            for i, target in enumerate(tasks)
                        },
                    }
                ),
            )
            if WANDB:
                run.log(metrics_dict)

    if log:
        import os

        torch.save(model.state_dict(), os.path.join(logger.root, "model.pt"))


# %% ---------------------------- RUN
start_run()
# %%
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

model = Model(num_tasks=num_tasks)
model.load_state_dict(torch.load("logs/nuclear/model.pt"))
from sklearn.decomposition import PCA
import numpy as np

n = 2
pca = PCA(n_components=n)

emb = model.embedding.weight.detach().cpu().numpy()
emb = pca.fit_transform(emb)
emb = np.hstack([emb, np.zeros_like(emb)])
fig, ax = plt.subplots(figsize=(15, 5))
c = plt.cm.viridis(np.linspace(0, 1, len(emb)))
plt.scatter(emb[:, 0], emb[:, 1], c=c, s=1)
for i, (x, y, *_) in enumerate(emb):
    ax.annotate(f"{i}", (x, y), fontsize=10, c=c[i])
plt.savefig(f"nuclear_embeddings{n}d.pdf")
# %%
