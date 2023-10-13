import wandb
import os
import torch
import random
import numpy as np

from utils.utils import get_transformations, build_model, setup_dataloader, setup_criterion
from vanilla_singletask_model.trainer import Trainer as VSTMT
from vanilla_multitask_model.trainer import Trainer as VMTMT
from multitask_w_seghead_model.trainer import Trainer as MTUNetSegMT
from torch.optim.lr_scheduler import ExponentialLR

# empty cuda cache
torch.cuda.empty_cache()

# TODO: To check if this helps with the memory issue
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# set up random seed for reproducibility
torch.manual_seed(2023)
random.seed(2023)
np.random.seed(2023)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.use_deterministic_algorithms(True)
torch.autograd.set_detect_anomaly(True)

def run_experiment():
        # set up augmentation
    data_transforms = get_transformations()

    # all changeable parameters should be specified here
    wandb.config.update({
        'batch_size': 2,
        'num_workers': 0,  # anything > 0 is giving issues -- why?
        'pin_memory': True,
        'num_epochs': 40,
        #-------------------------------------------------------------------------------------------------------------------------------------------------------------
        'loss': 'MT-dice-overlap_penalty-seghead-loss',     # 'MSE, 'SteeperMSE', 'CE', 'MT-dice-seghead-loss','MT-WSdice-seghead-loss', 'MT-dice-overlap_penalty-seghead-loss'
        'steeper_MSE_coeff': 10,
        'weighted_soft_dice_v1': 0.1,
        'overlap_loss_weight': 10,
        #-------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Model specifications
        'model_type': 'multitask_UNET_segmentation', # 'multitask_UNET_segmentation', 'vanilla-singletask'
        'model_name': 'vit_base',   # 'swin'/'vit' + '_'  + 'small', 'tiny', 'base' || 'resnet' + '18'/'50'
        'lr': 0.01,
        'weight_decay': 0.001,
        'gamma': 0.85,
        #-------------------------------------------------------------------------------------------------------------------------------------------------------------
        # This is for the singletask model
        'num_classes': [1, 1, 1],           # 4/5; 
        #-------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Dataset specifications
        'anatomy_part': ['Glans', 'Meatus', 'Shaft'],
        'overlap_bodyparts': ['Foreskin'], # ['Foreskin', 'Fingers'],
        #-------------------------------------------------------------------------------------------------------------------------------------------------------------
        # UNet model specifications
        'in_channels': 3,
        'out_channels': 3,
        'UNET features': [64, 128, 256, 512],
        #-------------------------------------------------------------------------------------------------------------------------------------------------------------
        'freeze': False,
        'freeze_until': 'layer3',
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),     # 'cuda'
        'pretrained': True,
        'data_transforms': data_transforms,
        'data_path': "/home/pedke/dataset/datasets/30-August-23_split - no_special_chars", 
        'csv_path': "/home/pedke/dataset/datasheets/FINAL Copy of Classification - no_special_chars.xlsx",
        'extra_train_datasets': ['train-2'],
        'save_weights': True,
        'debug': False
    })

    print(f'Device: {wandb.config["device"]}')
    # setup model
    models = build_model(wandb.config)

    # create train dataset and dataloaders
    train_loader, valid_loader, test_loader, extra_train_ds_loaders = setup_dataloader(wandb.config, data_transforms)

    # set up loss function
    criterion = setup_criterion(wandb.config)

    # create and run trainer
    if wandb.config['model_type'].startswith('vanilla-singletask'):

        # set up optimizer and scheduler and put models on device
        optimizers, schedulers = {}, {}
        for bodypart in wandb.config['anatomy_part']:
            models[bodypart] = models[bodypart].to(wandb.config['device'])
            optimizers[bodypart] = torch.optim.Adam(params=models[bodypart].parameters(), lr=wandb.config['lr'],
                                                    weight_decay=wandb.config['weight_decay'])
            schedulers[bodypart] = ExponentialLR(optimizers[bodypart], gamma=wandb.config['gamma'])


        VSTMT(config=wandb.config, models=models, criterion=criterion, optimizers=optimizers, schedulers=schedulers).run(
            train_loader, valid_loader, extra_train_ds_loaders
        )
    elif wandb.config['model_type'].startswith('vanilla-multitask'):
        models = models.to(wandb.config['device'])

        optimizers = torch.optim.Adam(params=models.parameters(), lr=wandb.config['lr'],
                                                weight_decay=wandb.config['weight_decay'])
        schedulers = ExponentialLR(optimizers, gamma=wandb.config['gamma'])

        VMTMT(config=wandb.config, models=models, criterion=criterion, optimizers=optimizers,schedulers=schedulers).run(
            train_loader, valid_loader, extra_train_ds_loaders
        )

    elif wandb.config['model_type'].startswith('multitask_UNET_segmentation'):
        models = models.to(wandb.config['device'])

        optimizers = torch.optim.Adam(params=models.parameters(), lr=wandb.config['lr'], weight_decay=wandb.config['weight_decay'])
        schedulers = ExponentialLR(optimizers, gamma=wandb.config['gamma'])

        MTUNetSegMT(config=wandb.config, models=models, criterion=criterion, optimizers=optimizers, schedulers=schedulers).run(
            train_loader, valid_loader, extra_train_ds_loaders
        )


if __name__ == '__main__':

    # init wandb
    wandb.init(project="dumb") # "new-loss-summation"

    # run experiment
    run_experiment()

    # save the model to wandb
    if wandb.config['save_weights']:
        wandb.save(os.path.join(wandb.run.dir, 'model.pth'))

    # explicitly end wandb
    wandb.finish()
