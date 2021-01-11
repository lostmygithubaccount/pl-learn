# imports
import gzip
import time
import torch
import mlflow
import random
import argparse

import numpy as np
import pytorch_lightning as pl

from torch import nn
from torch.nn import functional as F
from adlfs import AzureBlobFileSystem
from torch.utils.data import DataLoader
from sklearn.preprocessing import OneHotEncoder

# hacks
from azureml_env_adapter import set_environment_variables

# pytorch lightning data module
class AzureMLMNISTDataModule(pl.LightningModule):
    def __init__(self, batch_size: int = 10):
        super().__init__()
        self.batch_size = batch_size
        self.setup()

    def setup(self, stage=None):
        data_dir = "datasets/mnist"
        storage_options = {"account_name": "azuremlexamples"}
        fs = AzureBlobFileSystem(**storage_options)
        files = fs.ls(data_dir)

        train_len = 60000
        test_len = 10000

        for f in files:
            if "train-images" in f:
                self.X_train = self._read_images(gzip.open(fs.open(f)), train_len)
            elif "train-labels" in f:
                self.y_train = self._read_labels(gzip.open(fs.open(f)), train_len)
            elif "images" in f:
                self.X_test = self._read_images(gzip.open(fs.open(f)), test_len)
            elif "labels" in f:
                self.y_test = self._read_labels(gzip.open(fs.open(f)), test_len)

        self.ohe = OneHotEncoder().fit(self.y_train.reshape(-1, 1))

        self.mnist_train = list(
            zip(
                self.X_train, self.ohe.transform(self.y_train.reshape(-1, 1)).toarray(),
            )
        )
        self.mnist_test = list(
            zip(self.X_test, self.ohe.transform(self.y_test.reshape(-1, 1)).toarray(),)
        )

    def _read_images(self, f, images):
        image_size = 28

        f.read(16)  # magic

        buf = f.read(image_size * image_size * images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(images, image_size, image_size, 1)

        return data

    def _read_labels(self, f, labels):
        f.read(8)  # magic

        buf = f.read(1 * labels)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        return labels

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)


# pytorch lightning module
class MLPSystem(pl.LightningModule):
    def __init__(self, batch_size):
        super().__init__()

        self.batch_size = batch_size

        self.net = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
            nn.Softmax(),
        )

    def forward(self, x):
        x = self.net(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(self.batch_size, -1)
        y = y.view(self.batch_size, -1)
        y_hat = self.forward(x)
        loss = F.binary_cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(self.batch_size, -1)
        y = y.view(self.batch_size, -1)
        y_hat = self.forward(x)
        loss = F.binary_cross_entropy(y_hat, y)
        self.log("test_loss", loss)
        acc = (y_hat == y).sum() / len(y_hat)
        self.log("test_acc", acc)
        return {"loss": loss, "acc": acc}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# setup argparse
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--max_epochs", type=int, default=100)
parser.add_argument("--gpus", type=int, default=-1)
parser.add_argument("--accelerator", type=str, default="ddp")
parser.add_argument("--precision", type=int, default=16)
parser.add_argument("--seed", type=str, default=random.randint(0, 2 << 32))
args = parser.parse_args()

# hack
set_environment_variables()

# randomness
pl.seed_everything(args.seed)

# setup data
mnist = AzureMLMNISTDataModule(batch_size=args.batch_size)

# setup system
system = MLPSystem(batch_size=args.batch_size)

# setup trainer
trainer = pl.Trainer.from_argparse_args(args)

# fit system
t1 = time.time()
trainer.fit(system, mnist.train_dataloader())
t2 = time.time()
mlflow.log_metric("training_time", t2 - t1)

# test system
t1 = time.time()
trainer.test(system, mnist.test_dataloader())
t2 = time.time()
mlflow.log_metric("test_time", t2 - t1)
