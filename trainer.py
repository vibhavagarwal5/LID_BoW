import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from pytorch_lightning.metrics.functional import accuracy
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from dataset import BoWDataset, lang, Collatefn


class LangDetection(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.model = nn.Sequential(
            nn.Linear(hparams.vocab_len, 500),
            nn.ReLU(),
            nn.Linear(500, 300),
            nn.ReLU(),
            nn.Linear(300, 150),
            nn.ReLU(),
            nn.Linear(150, self.hparams.num_classes))
        self.accuracy = pl.metrics.Accuracy()

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.model(x)
        loss = F.cross_entropy(
            z.view(-1, self.hparams.num_classes), y.view(-1))
        self.log('train_loss', loss)
        self.accuracy(z.argmax(dim=1), y)
        return loss

    def training_epoch_end(self, outs):
        self.log('train_acc_epoch', self.accuracy.compute())

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.model(x)
        loss = F.cross_entropy(
            z.view(-1, self.hparams.num_classes), y.view(-1))
        self.log('val_loss', loss)
        self.log('val_acc_step', self.accuracy(z.argmax(dim=1), y))

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.model(x)
        loss = F.cross_entropy(
            z.view(-1, self.hparams.num_classes), y.view(-1))
        self.log('test_loss', loss)
        # acc = torch.eq(y, z.argmax(dim=1)).float().sum() / len(y)
        self.log('test_acc_step', self.accuracy(z.argmax(dim=1), y))

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
    train = DataLoader(train,
                       batch_size=hparams.batch_size,
                       num_workers=hparams.num_workers,
                       shuffle=True,
                       collate_fn=Collatefn())

    val = BoWDataset('val', hparams.data_path, hparams)
    val = DataLoader(val,
                     batch_size=hparams.batch_size,
                     num_workers=hparams.num_workers,
                     collate_fn=Collatefn()
                     )

    test = BoWDataset('test', hparams.data_path, hparams)
    test = DataLoader(test,
                      batch_size=hparams.batch_size,
                      num_workers=hparams.num_workers,
                      collate_fn=Collatefn())

    """
    Main
    """
    ae = LangDetection(hparams)
    trainer = pl.Trainer.from_argparse_args(
        hparams,
        gpus=hparams.num_gpus,
        max_epochs=hparams.n_epochs)
    trainer.fit(ae, train, val)
    trainer.test(test_dataloaders=test)


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--num_gpus", type=int,
                        default=1, help="No. of GPUs, if >1 then ddp used")
    parser.add_argument("--data_path", type=str,
                        default="./data", help="Path of the dataset")
    parser.add_argument("--batch_size", type=int,
                        default=128, help="Batch size for training")
    parser.add_argument("--num_workers", type=int,
                        default=32, help="Number of workers")
    parser.add_argument("--n_epochs", type=int, default=4,
                        help="Number of training epochs")
    parser.add_argument("--num_classes", type=int, default=len(lang),
                        help="Number of language classes to classify")
    parser.add_argument("--lr", type=float,
                        default=3e-5, help="Learning rate")

    # group the Trainer arguments together
    parser = pl.Trainer.add_argparse_args(
        parser.add_argument_group(title="pl.Trainer args")
    )

    hparams = parser.parse_args()
    return hparams


if __name__ == '__main__':
    hparams = get_args()
    main(hparams)
