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

# set up random seed for reproducibility
torch.manual_seed(2023)
random.seed(2023)
np.random.seed(2023)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.use_deterministic_algorithms(True)

def run_experiment():
        # set up augmentation
    data_transforms = get_transformations()

    # all changeable parameters should be specified here
    wandb.config.update({
        'batch_size': 2,
        'num_workers': 0,  # anything > 0 is giving issues -- why?
        'pin_memory': True,
        'num_epochs': 40,
        'loss': 'MT-indep-seghead-loss',     # 'MSE, 'SteeperMSE', 'CE', 'MT-indep-seghead-loss', 'MT-new-seghead-loss'
        'steeper_MSE_coeff': 10,
        'model_type': 'multitask_UNET_segmentation',
        'model_name': 'vit_base',   # 'swin'/'vit' + '_'  + 'small', 'tiny', 'base' || 'resnet' + '18'/'50'
        'lr': 0.01,
        'weight_decay': 0.001,
        'gamma': 0.85,
        'num_classes': [1, 1, 1],           # 4/5
        'anatomy_part': ['Glans', 'Meatus', 'Shaft'],
        'in_channels': 3,
        'out_channels': 1,
        'UNET features': [64, 128, 256, 512],
        'freeze': False,
        'freeze_until': 'layer3',
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),     # 'cuda'
        'pretrained': True,
        'data_transforms': data_transforms,
        'data_path': r"C:\Users\Parinita Edke\Desktop\Hypospadias-Project\Hypospadias-Data\datasets\19-April-23_split - no_special_chars",# '/home/mrizhko/hn_miccai/data/', 
        'csv_path': r"C:\Users\Parinita Edke\Desktop\Hypospadias-Project\Hypospadias-Data\datasheets\FINAL Copy of Classification - no_special_chars.xlsx",
        'save_weights': True,
        'debug': False
    })

    print(f'Device: {wandb.config["device"]}')
    # setup model
    models = build_model(wandb.config)

    # create train dataset and dataloaders
    train_loader, valid_loader, test_loader = setup_dataloader(wandb.config, data_transforms)

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
            train_loader, valid_loader
        )
    elif wandb.config['model_type'].startswith('vanilla-multitask'):
        models = models.to(wandb.config['device'])

        optimizers = torch.optim.Adam(params=models.parameters(), lr=wandb.config['lr'],
                                                weight_decay=wandb.config['weight_decay'])
        schedulers = ExponentialLR(optimizers, gamma=wandb.config['gamma'])

        VMTMT(config=wandb.config, models=models, criterion=criterion, optimizers=optimizers,
              schedulers=schedulers).run(
            train_loader, valid_loader
        )

    elif wandb.config['model_type'].startswith('multitask_UNET_segmentation'):
        models = models.to(wandb.config['device'])

        optimizers = torch.optim.Adam(params=models.parameters(), lr=wandb.config['lr'],
                                      weight_decay=wandb.config['weight_decay'])
        schedulers = ExponentialLR(optimizers, gamma=wandb.config['gamma'])

        MTUNetSegMT(config=wandb.config, models=models, criterion=criterion, optimizers=optimizers,
              schedulers=schedulers).run(
            train_loader, valid_loader
        )


if __name__ == '__main__':

    # init wandb
    wandb.init(project="multitask-UNET-segmentation-model")

    # run experiment
    run_experiment()

    # save the model to wandb
    if wandb.config['save_weights']:
        wandb.save(os.path.join(wandb.run.dir, 'model.pth'))

    # explicitly end wandb
    wandb.finish()
