from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.decomposition import PCA
import numpy as np
import torch
import os
from train import model
import tqdm
from sys import argv
plt.style.use("mplstyle.mplstyle")


def _get_metrics(log_dir):
    # Load Data
    metrics = np.loadtxt(
        os.path.join(log_dir, "metrics.csv"), delimiter=",", skiprows=1
    )
    return metrics.T


def _load_embeddings(log_dir, epoch):
    # Load Model
    model.load_state_dict(torch.load(os.path.join(log_dir, "models", f"{epoch}.pt")))
    return model.embedding.weight.detach().numpy()


def animate_embedddings(log_dir):
    # Load Data
    print(f"Loading {log_dir}...")
    epochs, train_loss, train_acc, val_loss, val_acc = _get_metrics(log_dir)
    epochs = epochs[::20]
    train_loss = train_loss[::20]
    train_acc = train_acc[::20]
    val_loss = val_loss[::20]
    val_acc = val_acc[::20]
    

    # PCA
    all_embeddings = []
    pca = PCA(n_components=2)
    # Load the last model to get the final PC's
    embeddings = _load_embeddings(log_dir, epochs[-1])
    pca.fit(embeddings)
    for epoch in epochs:
        embeddings = _load_embeddings(log_dir, epoch)
        all_embeddings.append(pca.transform(embeddings))
    all_embeddings = np.array(all_embeddings)
    orig_shape = all_embeddings.shape
    all_embeddings = all_embeddings.reshape(-1, all_embeddings.shape[-1])
    min_ = np.min(all_embeddings, axis=0)
    max_ = np.max(all_embeddings, axis=0)
    all_embeddings = (all_embeddings - min_) / (max_ - min_) * 2 - 1
    all_embeddings = all_embeddings.reshape(orig_shape)
    print(f"Loaded {len(all_embeddings)} embeddings")

    # Plot
    fig, ax = plt.subplots(dpi=300)
    # ax.set_xlim(min(all_embeddings[0][:, 0]), max(all_embeddings[0][:, 0]))
    # ax.set_ylim(min(all_embeddings[0][:, 1]), max(all_embeddings[0][:, 1]))
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ax.set_title("Embeddings")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()

    P = embeddings.shape[0]
    colors = plt.cm.viridis(np.linspace(0, 1, P))

    sc = ax.scatter([0] * P, [0] * P, c=colors, s=2)
    # annotate each embedding with its index
    an = [ax.annotate(i, (0, 0), color=colors[i], fontsize=6) for i in range(P)]
    metrics = ax.text(0.0, 0.975, "", transform=ax.transAxes, fontsize=6)

    pbar = tqdm.trange(len(epochs), desc="Plotting...", leave=False)

    def update(i):
        pbar.update()
        sc.set_offsets(all_embeddings[i])
        for j, txt in enumerate(an):
            txt.set_position(all_embeddings[i][j])
        msg = (
            f"Epoch: {epochs[i]} Train Loss: {train_loss[i]:.2f} "
            + f"Acc: {train_acc[i]:.0f} | "
            + f"Val Loss: {val_loss[i]:.2f} "
            + f"Acc: {val_acc[i]:.0f} "
        )
        metrics.set_text(msg)
        return sc, *an, metrics

    anim = FuncAnimation(fig, update, frames=len(epochs), blit=True)
    name = os.path.basename(log_dir)
    savefile = os.path.join(f"emb_{name}.mp4")
    anim.save(savefile, writer="ffmpeg", fps=len(epochs) // 10)
    print(f"Saved {savefile}\n")


def plot_metrics(log_dir):
    # Load Data
    print(f"Loading {log_dir}...")
    epochs, train_loss, train_acc, val_loss, val_acc = _get_metrics(log_dir)

    # Plot
    fig, ax = plt.subplots(2, 1, sharex=True, dpi=200)
    ax[0].plot(epochs, train_loss, label="Train")
    ax[0].plot(epochs, val_loss, label="Val")
    ax[0].set_ylabel("Loss")
    ax[0].legend()
    ax[0].set_yscale("log")
    ax[0].set_ylim(train_loss.min()*0.9, val_loss.max() * 1.)

    ax[1].plot(epochs, train_acc, label="Train")
    ax[1].plot(epochs, val_acc, label="Val")
    ax[1].set_ylabel("Accuracy")
    ax[1].set_xlabel("Epoch")
    ax[0].set_xscale("log")
    ax[1].legend()

    fig.tight_layout()
    name = os.path.basename(log_dir)
    savefile = os.path.join(f"metrics_{name}.jpg")
    fig.savefig(savefile)
    print(f"Saved {savefile}\n")

if __name__ == "__main__":
    if "NAME" in os.environ:
        logs = [os.environ["NAME"]]
    # if "--name" in argv:
    #     logs = [argv[argv.index("--name") + 1]]
    else:
        logs = os.listdir("log")
    if not os.path.exists("log") or len(logs) == 0:
        print("No logs found. Run train.py first.")
        exit()
    for log in logs:
        log_dir = os.path.join("log", log)
        if "--anim" in argv:
            animate_embedddings(log_dir)
        plot_metrics(log_dir)
