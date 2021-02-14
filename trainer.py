import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from pytorch_lightning.metrics.functional import accuracy
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from dataset import BoWDataset, lang


class LangDetection(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.model = nn.Sequential(
            nn.Linear(hparams.vocab_len, 300),
            nn.ReLU(),
            nn.Linear(300, 150),
            nn.ReLU(),
            nn.Linear(150, self.hparams.num_classes))
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.model(x)
        loss = F.cross_entropy(
            z.view(-1, self.hparams.num_classes), y.view(-1))
        self.log('train_loss', loss)
        acc = torch.eq(y, z.argmax(dim=1)).float().sum() / len(y)
        self.log('train_acc_step', acc, on_step=True,
                 on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.model(x)
        loss = F.cross_entropy(
            z.view(-1, self.hparams.num_classes), y.view(-1))
        self.log('val_loss', loss)
        acc = torch.eq(y, z.argmax(dim=1)).float().sum() / len(y)
        self.log('val_acc_step', acc, on_step=True,
                 on_epoch=False, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.model(x)
        loss = F.cross_entropy(
            z.view(-1, self.hparams.num_classes), y.view(-1))
        self.log('test_loss', loss)
        acc = torch.eq(y, z.argmax(dim=1)).float().sum() / len(y)
        self.log('test_acc_step', acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer


def main(hparams):
    pl.seed_everything(42)

    """
    Data
    """
    train = BoWDataset('train', hparams.data_path, hparams)
    hparams.vocab_len = len(train.vocab)
    train = DataLoader(train, batch_size=hparams.batch_size)

    val = BoWDataset('val', hparams.data_path, hparams)
    val = DataLoader(val, batch_size=hparams.batch_size)

    test = BoWDataset('test', hparams.data_path, hparams)
    test = DataLoader(test, batch_size=hparams.batch_size)

    """
    Main
    """
    ae = LangDetection(hparams)
    trainer = pl.Trainer(gpus=hparams.num_gpus,
                         max_epochs=hparams.n_epochs)
    trainer.fit(ae, train, val)

    trainer.test(test_dataloaders=test)


def get_args():
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--num_gpus", type=int,
                        default=1, help="No. of GPUs, if >1 then ddp used")
    parser.add_argument("--data_path", type=str,
                        default="./data", help="Path of the dataset")
    parser.add_argument("--batch_size", type=int,
                        default=4, help="Batch size for training")
    parser.add_argument("--n_epochs", type=int, default=4,
                        help="Number of training epochs")
    parser.add_argument("--num_classes", type=int, default=len(lang),
                        help="Number of language classes to classify")
    parser.add_argument("--lr", type=float,
                        default=1e-4, help="Learning rate")
    hparams = parser.parse_args()
    return hparams


if __name__ == '__main__':
    hparams = get_args()
    main(hparams)
