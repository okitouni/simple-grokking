import torch
import os
import shutil


class Logger:
    def __init__(self, name, metrics=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]):
        self.metrics = metrics

        self.root = os.path.join("logs", name)
        models = os.path.join(self.root, "models")
        os.makedirs(self.root, exist_ok=True)
        if os.path.exists(models): shutil.rmtree(models)
        os.makedirs(models, exist_ok=True)

        self.metrics_file = open(os.path.join(self.root, "metrics.csv"), "w")
        self.metrics_file.write(",".join(metrics) + "\n")
        self.model_file = os.path.join(models, "{}.pt")

    def log(self, model=None, **metrics):
        for metric in self.metrics:
            value = metrics.get(metric, -1)
            self.metrics_file.write(f"{value},")
        self.metrics_file.write("\n")

        if model is not None:
            epoch = metrics.get("epoch", "latest")
            torch.save(model.state_dict(), self.model_file.format(epoch))
