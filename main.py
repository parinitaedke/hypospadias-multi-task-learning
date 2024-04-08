# IMPORTS
import wandb
import os
import torch
import random
import numpy as np

from utils.utils import get_transformations, build_model, setup_dataloader, setup_criterion
from vanilla_singletask_model.trainer import Trainer as VSTMT
from vanilla_multitask_model.trainer import Trainer as VMTMT
from vanilla_multitask_model.component_trainer import Trainer as component_VMTMT
from multitask_w_seghead_model.trainer import Trainer as MTUNetSegMT
from multitask_w_seghead_model.component_trainer import Trainer as component_MTUNetSegMT
from multitask_w_attention_model.trainer import Trainer as AttentionTrainer
from segmentation_model.trainer import Trainer as SegmentationTrainer

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

# GLOBAL VARS
ENCODER = 'encoder'
DECODER = 'decoder'
CLASSIFICATION_HEADS = 'classification_heads'

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
        'loss': 'Dice',     # 'MSE, 'SteeperMSE', 'CE', 'MT-dice-seghead-loss','MT-WSdice-seghead-loss', 'MT-dice-overlap_penalty-seghead-loss', 'CE-Attention
        'steeper_MSE_coeff': 100,
        'weighted_soft_dice_v1': 0.1,
        'overlap_loss_weight': 10,
        #-------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Model specifications
        'model_type': 'segmentation', # 'attention', 'multitask_UNET_segmentation', 'vanilla-singletask', 'vanilla-multitask', 'component-multitask_UNET_segmentation'
        'model_name': 'resnet50',   # 'swin'/'vit' + '_'  + 'small', 'tiny', 'base' || 'resnet' + '18'/'50'
        'lr': 0.01,
        'weight_decay': 0.001,
        'gamma': 0.85,
        'hope_classification_heads': False,
        'gms_classification_heads': True,
        #-------------------------------------------------------------------------------------------------------------------------------------------------------------
        # This is for the singletask model
        'num_classes': [1, 1, 1],           # 4/5; 
        #-------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Dataset specifications
        'anatomy_part': ['Urethral Plate'], #, 'Meatus', 'Shaft'],
        'overlap_bodyparts': [],#['Foreskin'], # ['Foreskin', 'Fingers'],
        'hope_components': ['Position of meatus', 'Shape of meatus', 'Shape of glans', 'Shape of skin', 'Torsion'],
        #-------------------------------------------------------------------------------------------------------------------------------------------------------------
        # UNet model specifications
        'in_channels': 3,
        'out_channels': 1,
        'UNET features': [64, 128, 256, 512],
        'activation': 'sigmoid',
        'encoder_weights': 'imagenet',
        #-------------------------------------------------------------------------------------------------------------------------------------------------------------
        'freeze': False,
        'freeze_until': 'layer3',
        'device':  torch.device("cuda" if torch.cuda.is_available() else "cpu"),     # 'cuda'
        'pretrained': True,
        'data_transforms': data_transforms,
        'data_path': "/home/pedke/dataset/datasets/15-October-23_split - no_special_chars", 
        'csv_path': "/home/pedke/dataset/datasheets/FINAL Copy of Classification - no_special_chars.xlsx",
        'extra_train_datasets': [], # ['train-2'],
        'extra_val_datasets': [], #['val-2'],
        'save_weights': True,
        'debug': False
    })

    print(f'Device: {wandb.config["device"]}')
    # setup model
    models = build_model(wandb.config)

    # create train dataset and dataloaders
    train_loader, valid_loader, test_loader, extra_train_ds_loaders, extra_val_ds_loaders = setup_dataloader(wandb.config, data_transforms)

    # set up loss function
    criterion = setup_criterion(wandb.config, models)

    # create and run trainer
    if wandb.config['model_type'].startswith('vanilla-singletask'):

        # set up optimizer and scheduler and put models on device
        # creates an optimizer and scheduler for each body part
        optimizers, schedulers = {}, {}
        for bodypart in wandb.config['anatomy_part']:
            models[bodypart] = models[bodypart].to(wandb.config['device'])
            optimizers[bodypart] = torch.optim.Adam(params=models[bodypart].parameters(), lr=wandb.config['lr'],
                                                    weight_decay=wandb.config['weight_decay'])
            schedulers[bodypart] = ExponentialLR(optimizers[bodypart], gamma=wandb.config['gamma'])


        VSTMT(config=wandb.config, models=models, criterion=criterion, optimizers=optimizers, schedulers=schedulers).run(
            train_loader, valid_loader, extra_train_ds_loaders, extra_val_ds_loaders
        )
    elif wandb.config['model_type'].startswith('vanilla-multitask'):
        
        # set up optimizer and scheduler and put models on device
        # creates a single optimizer and scheduler for the model
        models = models.to(wandb.config['device'])

        optimizers = torch.optim.Adam(params=models.parameters(), lr=wandb.config['lr'],
                                                weight_decay=wandb.config['weight_decay'])
        schedulers = ExponentialLR(optimizers, gamma=wandb.config['gamma'])

        VMTMT(config=wandb.config, models=models, criterion=criterion, optimizers=optimizers,schedulers=schedulers).run(
            train_loader, valid_loader, extra_train_ds_loaders, extra_val_ds_loaders
        )
    
    elif wandb.config['model_type'].startswith('component-vanilla-multitask'):
        
        # set up optimizer and scheduler and put models on device
        # creates different optimizers and schedulers for each body part, allowing for model updates to occur separately
        optimizers, schedulers = {}, {}
        
        models[ENCODER] = models[ENCODER].to(wandb.config['device'])
        optimizers[ENCODER] = torch.optim.Adam(params=models[ENCODER].parameters(), lr=wandb.config['lr'],
                                                weight_decay=wandb.config['weight_decay'])
        schedulers[ENCODER] = ExponentialLR(optimizers[ENCODER], gamma=wandb.config['gamma'])
        
        if wandb.config['hope_classification_heads'] and wandb.config['gms_classification_heads']:
            bodypart_lst = wandb.config['anatomy_part'] + wandb.config['hope_components']
        elif not wandb.config['hope_classification_heads'] and wandb.config['gms_classification_heads']:
            bodypart_lst = wandb.config['anatomy_part']
        elif wandb.config['hope_classification_heads'] and not wandb.config['gms_classification_heads']:
            bodypart_lst = wandb.config['hope_components']
            
        optimizers[CLASSIFICATION_HEADS] = {}
        schedulers[CLASSIFICATION_HEADS] = {}
        
        for subcomponent in bodypart_lst:
            models[CLASSIFICATION_HEADS][subcomponent] = models[CLASSIFICATION_HEADS][subcomponent].to(wandb.config['device'])
            optimizers[CLASSIFICATION_HEADS][subcomponent] = torch.optim.Adam(params=models[CLASSIFICATION_HEADS][subcomponent].parameters(), lr=wandb.config['lr'],
                                                    weight_decay=wandb.config['weight_decay'])
            schedulers[CLASSIFICATION_HEADS][subcomponent] = ExponentialLR(optimizers[CLASSIFICATION_HEADS][subcomponent], gamma=wandb.config['gamma'])
            
            
        component_VMTMT(config=wandb.config, models=models, criterion=criterion, optimizers=optimizers,schedulers=schedulers, bodypart_lst=bodypart_lst).run(
            train_loader, valid_loader, extra_train_ds_loaders, extra_val_ds_loaders
        )

    elif wandb.config['model_type'].startswith('multitask_UNET_segmentation'):
        
        # set up optimizer and scheduler and put models on device
        # creates an optimizer and scheduler for each body part + UNET part
        models = models.to(wandb.config['device'])

        optimizers = torch.optim.Adam(params=models.parameters(), lr=wandb.config['lr'], weight_decay=wandb.config['weight_decay'])
        schedulers = ExponentialLR(optimizers, gamma=wandb.config['gamma'])

        MTUNetSegMT(config=wandb.config, models=models, criterion=criterion, optimizers=optimizers, schedulers=schedulers).run(
            train_loader, valid_loader, extra_train_ds_loaders, extra_val_ds_loaders
        )
    
    elif wandb.config['model_type'].startswith('component-multitask_UNET_segmentation'):
        
        # set up optimizer and scheduler and put models on device
        # creates different optimizers and schedulers for each body part + UNET part, allowing for model updates to occur separately
        optimizers, schedulers = {}, {}
        for component in [ENCODER, DECODER]: #, 'classification_heads']:
            models[component] = models[component].to(wandb.config['device'])
            optimizers[component] = torch.optim.Adam(params=models[component].parameters(), lr=wandb.config['lr'],
                                                    weight_decay=wandb.config['weight_decay'])
            schedulers[component] = ExponentialLR(optimizers[component], gamma=wandb.config['gamma'])
        
        if wandb.config['hope_classification_heads'] and wandb.config['gms_classification_heads']:
            bodypart_lst = wandb.config['anatomy_part'] + wandb.config['hope_components']
        elif not wandb.config['hope_classification_heads'] and wandb.config['gms_classification_heads']:
            bodypart_lst = wandb.config['anatomy_part']
        elif wandb.config['hope_classification_heads'] and not wandb.config['gms_classification_heads']:
            bodypart_lst = wandb.config['hope_components']
            
        optimizers[CLASSIFICATION_HEADS] = {}
        schedulers[CLASSIFICATION_HEADS] = {}
        for subcomponent in bodypart_lst:
            models[CLASSIFICATION_HEADS][subcomponent] = models[CLASSIFICATION_HEADS][subcomponent].to(wandb.config['device'])
            optimizers[CLASSIFICATION_HEADS][subcomponent] = torch.optim.Adam(params=models[CLASSIFICATION_HEADS][subcomponent].parameters(), lr=wandb.config['lr'],
                                                    weight_decay=wandb.config['weight_decay'])
            schedulers[CLASSIFICATION_HEADS][subcomponent] = ExponentialLR(optimizers[CLASSIFICATION_HEADS][subcomponent], gamma=wandb.config['gamma'])
            
            
        component_MTUNetSegMT(config=wandb.config, models=models, criterion=criterion, optimizers=optimizers,schedulers=schedulers, bodypart_lst=bodypart_lst).run(
            train_loader, valid_loader, extra_train_ds_loaders, extra_val_ds_loaders
        )
    
    elif wandb.config['model_type'].startswith('attention'):
        
        # set up optimizer and scheduler and put models on device
        # creates an optimizer and scheduler for each body part
        optimizers, schedulers = {}, {}
        for bodypart in wandb.config['anatomy_part']:
            models[bodypart] = models[bodypart].to(wandb.config['device'])
            optimizers[bodypart] = torch.optim.Adam(params=models[bodypart].parameters(), lr=wandb.config['lr'],
                                                    weight_decay=wandb.config['weight_decay'])
            schedulers[bodypart] = ExponentialLR(optimizers[bodypart], gamma=wandb.config['gamma'])


        AttentionTrainer(config=wandb.config, models=models, criterion=criterion, optimizers=optimizers, schedulers=schedulers).run(
            train_loader, valid_loader, extra_train_ds_loaders, extra_val_ds_loaders
        )
    
    elif wandb.config['model_type'].startswith('segmentation'):
        
        # set up optimizer and scheduler and put models on device
        # creates an optimizer and scheduler for each body part
        optimizers, schedulers = {}, {}
        for bodypart in wandb.config['anatomy_part']:
            models[bodypart] = models[bodypart].to(wandb.config['device'])
            optimizers[bodypart] = torch.optim.Adam(params=models[bodypart].parameters(), lr=wandb.config['lr'],
                                                    weight_decay=wandb.config['weight_decay'])
            schedulers[bodypart] = ExponentialLR(optimizers[bodypart], gamma=wandb.config['gamma'])

        SegmentationTrainer(config=wandb.config, models=models, criterion=criterion, optimizers=optimizers, schedulers=schedulers).run(
            train_loader, valid_loader, extra_train_ds_loaders, extra_val_ds_loaders
        )

        
        


if __name__ == '__main__':

    # init wandb -- project var referes to wandb project to store run in
    wandb.init(project="segmentation") # "segmentation", "ce-attention-model-softmax", "SUBcomponent-vanilla-multitask-model", "SUBcomponent-multitask-UNET-segmentation-model", "new-loss-summation", "overlap-multitask-UNET-segmentation-model", "vanilla-singletask-model", "vanilla-multitask-model", "component-overlap-multitask-UNET-segmentation-model"

    # run experiment
    run_experiment()

    # save the model to wandb
    if wandb.config['save_weights']:
        wandb.save(os.path.join(wandb.run.dir, 'model.pth'))

    # explicitly end wandb
    wandb.finish()