# get_transformations, build_model, setup_dataloader, setup_criterion
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm
import wandb
import pandas as pd
import albumentations as A
import albumentations.pytorch as Apy

# from loss import SteeperMSELoss, DiceLoss, InfluenceSegmentationLoss
from .loss import SteeperMSELoss, DiceLoss, WeightedSoftDiceLoss, InfluenceSegmentationLoss
from .dataset import HypospadiasDataset

from vanilla_multitask_model.model import Vanilla_Multitask_Model
from multitask_w_seghead_model.model import Vanilla_Multitask_UNET_Segmentation_Model

from multitask_w_seghead_model.component_model import Encoder, Decoder, ClassificationHeads

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state.state_dict(), filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_transformations():
    data_transforms = {
        'train': A.Compose([
            # A.PadIfNeeded(min_height=224, min_width=224),
            # A.RandomCrop(224, 224),
            A.Resize(224, 224), # (224, 224)
            A.Rotate(limit=30),
            # A.RandomResizedCrop(224, 224, ratio=(1.0, 1.0), scale=(0.9, 1.0)),
            A.HorizontalFlip(),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            Apy.transforms.ToTensorV2()
        ]),
        'valid': A.Compose([
            A.Resize(224, 224), # (224, 224)
            # A.Resize(256, 256),
            # A.CenterCrop(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.pytorch.transforms.ToTensorV2()
        ]),
    }

    return data_transforms

def build_model(config):
    
    if config['model_type'].startswith('vanilla-singletask'):

        # set up model name
        if config['model_name'].startswith('vit'):
            model_name = f"{config['model_name']}_patch16_224"
        elif config['model_name'].startswith('resnet'):
            model_name = config['model_name']
        else:
            raise NotImplementedError('Unknown model')

        # create model(s) for multiple body parts
        models = {}
        for i, body_part in enumerate(config['anatomy_part']):
            model = timm.create_model(model_name, pretrained=config['pretrained'], num_classes=config['num_classes'][i])

            # load weights
            # if config['weights_path']:
            #     model_path = wandb.restore('model.pth', run_path=config['weights_path'])
            #     model.load_state_dict(torch.load(model_path.name))

            # freeze model
            if config['freeze']:
                submodules = [n for n, _ in model.named_children()]
                timm.freeze(model, submodules[:submodules.index(config['freeze_until']) + 1])
            
            models[body_part] = model

    elif config['model_type'].startswith('vanilla-multitask'):
        models = Vanilla_Multitask_Model(config=config)

    elif config['model_type'].startswith('multitask_UNET_segmentation'):
        models = Vanilla_Multitask_UNET_Segmentation_Model(config=config)
    elif config['model_type'].startswith('component-multitask_UNET_segmentation'):
        models = {}
        
        models['encoder'] = Encoder(config=config)
        models['decoder'] = Decoder(config=config)
        models['classification_heads'] = ClassificationHeads(config=config)

    return models

def setup_dataloader(config, transforms):
    score_df = pd.read_excel(config["csv_path"],skiprows=1)
    score_df.rename(columns={score_df.columns[0]: "Image Name"}, inplace=True)
    score_df.drop(score_df.columns[[1, 18]], axis=1, inplace=True)

    train_ds = HypospadiasDataset(config=config, score_df=score_df, preprocessing=transforms['train'], dataset_type='train')
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], num_workers=config["num_workers"], pin_memory=config["pin_memory"], shuffle=True)
    
    extra_train_ds_loaders = []
    if any(config["extra_train_datasets"]):
        for extra_train_ds_name in config["extra_train_datasets"]:
            extra_train_ds = HypospadiasDataset(config=config, score_df=score_df, preprocessing=transforms['train'], dataset_type=extra_train_ds_name, include_masks=False)
            extra_train_loader = DataLoader(extra_train_ds, batch_size=config["batch_size"], num_workers=config["num_workers"], pin_memory=config["pin_memory"], shuffle=True)
            
            extra_train_ds_loaders.append(extra_train_loader)

    val_ds = HypospadiasDataset(config=config, score_df=score_df, preprocessing=transforms['valid'], dataset_type='val')
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], num_workers=config["num_workers"], pin_memory=config["pin_memory"], shuffle=True)
    
    extra_val_ds_loaders = []
    if any(config["extra_val_datasets"]):
        for extra_val_ds_name in config["extra_val_datasets"]:
            extra_val_ds = HypospadiasDataset(config=config, score_df=score_df, preprocessing=transforms['valid'], dataset_type=extra_val_ds_name, include_masks=False)
            extra_val_loader = DataLoader(extra_val_ds, batch_size=config["batch_size"], num_workers=config["num_workers"], pin_memory=config["pin_memory"], shuffle=True)
            
            extra_val_ds_loaders.append(extra_val_loader)
            

    test_ds = HypospadiasDataset(config=config, score_df=score_df, preprocessing=transforms['valid'], dataset_type='test')
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], num_workers=config["num_workers"], pin_memory=config["pin_memory"], shuffle=True)

    return train_loader, val_loader, test_loader, extra_train_ds_loaders, extra_val_ds_loaders

def setup_criterion(config):
    if config['loss'] == 'MSE':
        # criterion = BCEWithLogitsLoss()
        criterion = nn.MSELoss()

    elif config['loss'] == 'SteeperMSE':
        criterion = SteeperMSELoss(coefficient=config['steeper_MSE_coeff'])

    elif config['loss'] == 'CE':
        criterion = nn.CrossEntropyLoss() 

    elif config['loss'] == 'MT-dice-seghead-loss':
        # criterion1 = BCEWithLogitsLoss()
        criterion1 = DiceLoss()
        criterion2 = nn.MSELoss()

        criterion = [criterion1, criterion2]
    
    elif config['loss'] == 'MT-WSdice-seghead-loss':
        criterion1 = WeightedSoftDiceLoss(v1=config['weighted_soft_dice_v1'])
        criterion2 = nn.MSELoss()
        
        criterion = [criterion1, criterion2]

    elif config['loss'] == 'MT-dice-overlap_penalty-seghead-loss':
        criterion1 = DiceLoss()
        criterion2 = nn.MSELoss()
        criterion3 = InfluenceSegmentationLoss()
        criterion = [criterion1, criterion2, criterion3]

    else:
        raise NotImplementedError(f'Unknown loss')

    return criterion