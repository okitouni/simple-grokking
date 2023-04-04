from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.decomposition import PCA
import numpy as np
import torch
import os
from train import model
import tqdm


def _get_metrics(log_dir, skip=1):
    # Load Data
    metrics = np.loadtxt(
        os.path.join(log_dir, "metrics.csv"), delimiter=",", skiprows=1
    )
    epochs = metrics[::skip, 0].astype(int)
    train_loss = metrics[::skip, 1]
    train_acc = metrics[::skip, 2]
    val_loss = metrics[::skip, 3]
    val_acc = metrics[::skip, 4]
    return epochs, train_loss, train_acc, val_loss, val_acc


def _load_embeddings(log_dir, epoch):
    # Load Model
    model.load_state_dict(torch.load(os.path.join(log_dir, "models", f"{epoch}.pt")))
    return model.embedding.weight.detach().numpy()


def animate_embedddings(log_dir):
    # Load Data
    print(f"Loading {log_dir}...")
    epochs, train_loss, train_acc, val_loss, val_acc = _get_metrics(log_dir, skip=100)

    # PCA
    all_embeddings = []
    pca = PCA(n_components=2)
    # Load the last model to get the final PC's
    embeddings = _load_embeddings(log_dir, epochs[-1])
    pca.fit(embeddings)
    for epoch in epochs:
        embeddings = _load_embeddings(log_dir, epoch)
        all_embeddings.append(pca.transform(embeddings))
    print(f"Loaded {len(all_embeddings)} embeddings")

    # Plot
    fig, ax = plt.subplots(dpi=200)
    ax.set_xlim(min(all_embeddings[0][:, 0]), max(all_embeddings[0][:, 0]))
    ax.set_ylim(min(all_embeddings[0][:, 1]), max(all_embeddings[0][:, 1]))
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


if __name__ == "__main__":
    logs = os.listdir("log")
    if not os.path.exists("log") or len(logs) == 0:
        print("No logs found. Run train.py first.")
        exit()

    for log in logs:
        log_dir = os.path.join("log", log)
        animate_embedddings(log_dir)
