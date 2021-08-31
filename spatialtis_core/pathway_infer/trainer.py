import logging

import pytorch_lightning as pl
import torch

from .model import GCNG

logging.getLogger("lightning").setLevel(0)


def set_gpus():
    cuda = torch.cuda.is_available()
    if not cuda:
        return None
    cuda_count = torch.cuda.device_count()
    return cuda_count


def GCNG_model(train_loader, test_loader, lr=1e-4, gpus=None, max_epochs=10, random_seed=0):
    if gpus is None:
        gpus = set_gpus()

    gc = GCNG(lr=lr)
    pl.seed_everything(random_seed, workers=True)
    trainer = pl.Trainer(gpus=gpus,
                         max_epochs=max_epochs,
                         deterministic=True,
                         progress_bar_refresh_rate=0,
                         weights_summary=None)
    trainer.fit(gc, train_loader)
    trainer.test(dataloaders=test_loader, verbose=False)
    return gc
