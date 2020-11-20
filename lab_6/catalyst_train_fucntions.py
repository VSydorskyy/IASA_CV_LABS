import shutil
import os
import warnings

from typing import Callable, Optional
from os.path import join as pjoin

import pandas as pd
import numpy as np
import torch

from catalyst.dl import SupervisedRunner, Runner


def catalyst_training(
    train_df: pd.DataFrame, 
    val_df: pd.DataFrame, 
    exp_name: str, 
    train_dataset_class: torch.utils.data.Dataset,
    val_dataset_class: torch.utils.data.Dataset,
    train_dataset_config: dict,
    val_dataset_config: dict,
    train_dataloader_config: dict,
    val_dataloader_config: dict,
    nn_model_class: torch.nn.Module,
    nn_model_config: dict,
    optimizer_init: Callable,
    scheduler_init: Callable,
    criterion: torch.nn.Module,
    n_epochs: int,
    distributed: bool,
    catalyst_callbacks: Callable,
    main_metric: str,
    minimize_metric: bool,
    device: str,
    delete_logdir_folder: bool = True
):
    train_dataset = train_dataset_class(
        df=train_df,
        **train_dataset_config
    )
    val_dataset = val_dataset_class(
        df=val_df,
        **val_dataset_config
    )

    loaders = {
        'train':torch.utils.data.DataLoader(
            train_dataset,
            **train_dataloader_config
        ),
        'valid':torch.utils.data.DataLoader(
            val_dataset,
            **val_dataloader_config
        )
    }
    
    model = nn_model_class(
        device=device, 
        **nn_model_config
    )

    print(model)

    if distributed:
        model = torch.nn.DataParallel(model)

    optimizer = optimizer_init(model)
    scheduler = scheduler_init(optimizer)
    
    runner = SupervisedRunner()

    if os.path.exists(exp_name):
        warnings.warn(f"Logdir {exp_name} exists. Deleting!")
        shutil.rmtree(exp_name)

    runner.train(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        loaders=loaders,
        logdir=exp_name,
        num_epochs=n_epochs,
        verbose=True,
        load_best_on_end=True,
        main_metric=main_metric,
        minimize_metric=minimize_metric,
        callbacks=catalyst_callbacks() # We need to call this to make unique objects 
    )                                  # for each fold
    
    best_chkp = torch.load(pjoin(exp_name, 'checkpoints/best_full.pth'), map_location='cpu')

    if delete_logdir_folder:
        shutil.rmtree(exp_name)
    
    return best_chkp

def catalyst_inference(
    runner: Runner,
    dataset_class: torch.utils.data.Dataset,
    dataset_config: dict,
    dataloader_config: dict,
    nn_model_class: torch.nn.Module,
    nn_model_config: dict,
    checkpoint_path: str,
    device: str
):
    dataset = dataset_class(
        **dataset_config
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        **dataloader_config
    )
    model = nn_model_class(
        device=device, 
        **nn_model_config
    )
    prediction = np.concatenate([el['logits'].detach().cpu().numpy() for el in runner.predict_loader(
            loader=loader, 
            model=model,
            resume=checkpoint_path
        )])

    return prediction