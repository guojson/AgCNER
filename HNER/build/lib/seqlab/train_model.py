import pickle

import torch
from .model import BertWordCRF
import pytorch_lightning as pl
from seqeval.metrics import f1_score
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import argparse


class ModelTrainer(pl.LightningModule):

    def __init__(self, config):

        super().__init__()

        self.model = BertWordCRF(tag_to_id=config.tag_to_id,
                                 model_name=config.model_name, tag_format=config.tag_format, word_encoder=config.word_encoder, mode=config.mode)

        self.save_hyperparameters()

        self.config = config

    def training_step(self, batch, batch_idx):

        x = batch
        loss = self.model(x, compute_loss=True)['loss']
        self.log('train_loss', loss)

        return loss

    def validation_step(self, val_batch, batch_idx):

        x = val_batch

        prediction = self.model.predict(x)

        f1_micro_base = f1_score(
            x['iob_labels'], prediction, average="micro")

        self.log('f1_score', f1_micro_base, prog_bar=True)

    def train_dataloader(self):
        with open(self.config.data_path, 'rb') as f:
            train_data = pickle.load(f)['train']
        return self.model.data_processor.create_dataloader(train_data, batch_size=self.config.train_batch_size, num_workers=self.config.num_workers, shuffle=True)

    def val_dataloader(self):
        with open(self.config.data_path, 'rb') as f:
            dev_data = pickle.load(f)['dev']
        return self.model.data_processor.create_dataloader(dev_data, batch_size=self.config.val_batch_size, num_workers=self.config.num_workers, shuffle=False)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.config.learning_rate)
        return optimizer


def train_model(config):

    lightning_model = ModelTrainer(config)

    model_ckp = ModelCheckpoint(dirpath=config.dirpath, save_top_k=1,
                                monitor='f1_score', mode='max', filename=config.name)

    early_stop = EarlyStopping(patience=5, monitor='f1_score', mode='max')

    trainer = pl.Trainer(gpus=1, precision=32, max_epochs=config.max_epoch, callbacks=[
                         model_ckp, early_stop], check_val_every_n_epoch=config.check_val_every_n_epoch, limit_val_batches=config.limit_val_batches, val_check_interval=config.val_check_interval)

    trainer.fit(lightning_model)


class ConfigClass(object):
    def __init__(self, **kwargs):
        self.__dict__ = dict(kwargs)


def create_config(name, dirpath, data_path, model_name, word_encoder="transformer", mode="word", device="cuda", train_batch_size=8, val_batch_size=8, num_workers=1, learning_rate=2e-5, max_epoch=5, tag_format='BIO', check_val_every_n_epoch=1, limit_train_batches=1.0, limit_val_batches=1.0, val_check_interval=1.0):

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    tag_to_id = data['tag_to_id']

    cfg = ConfigClass(tag_to_id=tag_to_id, name=name, dirpath=dirpath, data_path=data_path, word_encoder=word_encoder, mode=mode, model_name=model_name, device=device, train_batch_size=train_batch_size, val_batch_size=val_batch_size, num_workers=num_workers,
                      learning_rate=learning_rate, max_epoch=max_epoch, tag_format=tag_format, check_val_every_n_epoch=check_val_every_n_epoch, limit_train_batches=limit_train_batches, limit_val_batches=limit_val_batches, val_check_interval=val_check_interval)

    return cfg
