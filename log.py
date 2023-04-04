import torch
import os
import shutil


class Logger:
    def __init__(self, name):
        root = os.path.join("log", name)
        models = os.path.join(root, "models")
        os.makedirs(root, exist_ok=True)
        if os.path.exists(models): shutil.rmtree(models)
        os.makedirs(models, exist_ok=True)

        self.metrics_file = open(os.path.join(root, "metrics.csv"), "w")
        self.metrics_file.write("epoch,train_loss,train_acc,val_loss,val_acc\n")
        self.model_file = os.path.join(models, "{}.pt")

    def log(self, model=None, **metrics):
        epoch = metrics.get("epoch", -1)
        train_loss = metrics.get("train_loss", -1)
        train_acc = metrics.get("train_acc", -1)
        val_loss = metrics.get("val_loss", -1)
        val_acc = metrics.get("val_acc", -1)

        self.metrics_file.write(f"{epoch},{train_loss},{train_acc},{val_loss},{val_acc}\n")

        if model is not None:
            torch.save(model.state_dict(), self.model_file.format(epoch))
