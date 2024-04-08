# IMPORTS
import numpy as np
import pandas as pd
import wandb
from tqdm import tqdm
import time
import os
import torch
import torch.nn.functional as F
from torchmetrics import MetricCollection
from torchmetrics import Dice, F1Score, Recall, Precision
import torchvision

from utils.utils import save_checkpoint

from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path


class Trainer:
    def __init__(self, config, models, criterion, optimizers, schedulers):
        self.config = config
        self.models = models
        self.criterion = criterion
        self.optimizers = optimizers
        self.schedulers = schedulers

        self.device = config['device']
        self.global_step = 0
        self.global_epoch = 0
        self.best_auroc = 0

        self.metrics = {}
        for bodypart in self.config['anatomy_part']:
            self.metrics[bodypart] = MetricCollection([Dice(),
                                                       #F1Score(),
                                                    #    Recall(),
                                                    #    Precision(),
                                                    ])

        self.losses = []


    def update_metrics(self, mask_pred, mask, loss=[]):
        
        loss = [item.cpu() for item in loss]
        self.losses.append(loss)
        
        # move variables to cpu
        mask_pred = mask_pred.detach().cpu()
        mask = mask.detach().cpu()

        # update metrics
        for i, bodypart in enumerate(self.config['anatomy_part']):
            self.metrics[bodypart](mask_pred, mask)


    def log_metrics(self, train):

        mean_bodypart_losses = np.asarray(self.losses).mean(axis=0)

        # log loss and calculate metrics over all batches
        for i, bodypart in enumerate(self.config['anatomy_part']):

            wandb.log({f"{train}_{bodypart}_loss": mean_bodypart_losses[i], 'epoch': self.global_epoch})

            metrics = self.metrics[bodypart].compute()

            # log metrics
            for metric in ['Dice']: #, 'F1Score', 'Recall', 'Precision']:
                wandb.log({f"{train}_{bodypart}_{metric}": metrics[metric], 'epoch': self.global_epoch})


    def reset_metrics(self):
        self.losses = []
        
        for bodypart in self.config['anatomy_part']:
            self.metrics[bodypart].reset()


    def train_epoch(self, data_loader, include_mask=True, overlap_mask_penalty=False):

        # set model to train mode
        # set model to train mode
        for bodypart in self.config['anatomy_part']:
            self.models[bodypart].train()
        
        for i, batch in enumerate(tqdm(data_loader)):

            names, X, labels,  mask = batch
                
            # move everything to cuda
            X = X.to(self.device)
            mask = mask.to(self.device)
            
            for j, bodypart in enumerate(self.config['anatomy_part']):
                
                # segmentation mask pred            
                pred = self.models[bodypart](X)
                mask_pred = pred[0]
                seg_losses = []

                lst_seg_pred = torch.split(mask_pred, 1, dim=1)  # Splits on the channel dimension
                
                # calculating the segmentation loss per mask type
                lst_gt_mask = torch.split(mask, 1, dim=1)  # Splits on the channel dimension
                
                # assert len(lst_seg_pred) == len(lst_gt_mask)
                
                for i in range(len(lst_seg_pred)):
                    seg_loss = self.criterion(lst_seg_pred[i], lst_gt_mask[i])
                    seg_losses.append(seg_loss)

                # calculating the segmentation loss for all masks
                total_segmentation_loss = torch.sum(torch.stack(seg_losses), dim=0)
                
                # # TODO: Add foreskin overlap penalty
                # if overlap_mask_penalty:
                #     overlap_loss = self.config['overlap_loss_weight'] * self.criterion[1](lst_gt_mask[len(lst_seg_pred):], lst_seg_pred[:len(lst_seg_pred)])
                    
                #     # total_loss += overlap_loss
                #     total_segmentation_loss += overlap_loss

                # calculate loss and optimize model
                self.models[bodypart].zero_grad()
                total_segmentation_loss.backward()
                self.optimizers[bodypart].step()
                
                self.global_step += 1

                # log batch loss
                wandb.log({f'{bodypart}_batch_segmentation_loss': total_segmentation_loss.item(), 'step': self.global_step})

                # save the segmentation maps per epoch to see how they evolve
                for i in range(len(lst_seg_pred)):
                    batch_split_pred = torch.split(lst_seg_pred[i], 1, dim=0)  # split by batches to get msk prediction for each sample for a specific bodypart
                    
                    if include_mask:
                        batch_split_gt = torch.split(lst_gt_mask[i], 1, dim=0)
                    
                    for b_, name in enumerate(names):
                        path = f"{self.saved_img_preds_dir}/Train/{name}/{self.config['anatomy_part'][i]}"
                        Path(path).mkdir(parents=True, exist_ok=True)
                        torchvision.utils.save_image(batch_split_pred[b_].sigmoid(), f"{path}/pred-epoch={self.global_epoch}.png")
                        
                        if include_mask:
                            torchvision.utils.save_image(batch_split_gt[b_].float(), f"{path}/gt.png")
            
            self.global_step += 1


        # log learning rate and update learning rate
        for bodypart in self.config['anatomy_part']:
            wandb.log({f'{bodypart}_learning_rate': self.optimizers[bodypart].param_groups[0]['lr'], 'epoch': self.global_epoch})
            self.schedulers[bodypart].step()


    def validate_epoch(self, data_loader, data_mode='val', include_mask=True):
        
        # set model to evaluation mode
        for bodypart in self.config['anatomy_part']:
            self.models[bodypart].eval()

        for i, batch in enumerate(tqdm(data_loader)):
            
            names, X, labels, mask = batch
            
            # move everything to cuda
            X = X.to(self.device)
            mask = mask.to(self.device)

            seg_losses = []
            for j, bodypart in enumerate(self.config['anatomy_part']):
                with torch.no_grad(): 
                    # calculate segmentation mask pred
                    pred = self.models[bodypart](X)
                    mask_pred = pred[0]
                    

                    # calculating the segmentation loss per mask type
                    lst_seg_pred = torch.split(mask_pred, 1, dim=1)  # splits on the channel dimension
                    
                    lst_gt_mask = torch.split(mask, 1, dim=1)  # splits on the channel dimension

                    # assert len(lst_seg_pred) == len(lst_gt_mask)

                    for i in range(len(lst_seg_pred)):
                        seg_loss = self.criterion(lst_seg_pred[i], lst_gt_mask[i])
                        seg_losses.append(seg_loss)
                        


                # save the segmentation maps per epoch to see how they evolve
                for i in range(len(lst_seg_pred)):
                    batch_split_pred = torch.split(lst_seg_pred[i], 1, dim=0)  # split by batches to get msk prediction for each sample for a specific bodypart
                    if include_mask:
                        batch_split_gt = torch.split(lst_gt_mask[i], 1, dim=0)
                    
                    for b_, name in enumerate(names):
                        
                        path = f"{self.saved_img_preds_dir}/Val/{data_mode}/{name}/{self.config['anatomy_part'][i]}"
                        Path(path).mkdir(parents=True, exist_ok=True)
                        torchvision.utils.save_image(batch_split_pred[b_].sigmoid(), f"{path}/pred-epoch={self.global_epoch}.png")
                        if include_mask:
                            torchvision.utils.save_image(batch_split_gt[b_].float(), f"{path}/gt.png")    
                

            # add loss to metrics
            self.update_metrics(mask_pred=mask_pred, mask=mask, loss=seg_losses)

    def run(self, train_loader, valid_loader, extra_train_ds_loaders, extra_val_ds_loaders):
        # TODO : EDIT
        since = time.time()

        # Make saved preds and checkpoints directory
        self.saved_preds_dir = f'{wandb.run.dir}/saved_preds'
        self.model_checkpoints_dir = f'{wandb.run.dir}/model_checkpoints'
        self.saved_img_preds_dir = f'{wandb.run.dir}/saved_img_preds'

        Path(self.saved_preds_dir).mkdir(parents=True, exist_ok=True)
        Path(self.model_checkpoints_dir).mkdir(parents=True, exist_ok=True)
        Path(f'{self.saved_img_preds_dir}/Train').mkdir(parents=True, exist_ok=True)
        Path(f'{self.saved_img_preds_dir}/Val').mkdir(parents=True, exist_ok=True)

        for epoch in range(self.config['num_epochs']):
            print('Epoch:', epoch)

            # train epoch
            self.train_epoch(train_loader, overlap_mask_penalty=any(self.config['overlap_bodyparts']))
            
            for extra_train_loader in extra_train_ds_loaders:
                self.train_epoch(extra_train_loader, include_mask=False)

            # validate and log on train data
            self.validate_epoch(train_loader, data_mode='train')
            
            for extra_train_loader in extra_train_ds_loaders:
                self.validate_epoch(extra_train_loader, data_mode='train', include_mask=False)

            self.log_metrics(train='train')
            self.reset_metrics()

            # validate and log on valid data
            self.validate_epoch(valid_loader, data_mode='val')
            
            for extra_val_loader in extra_val_ds_loaders:
                self.validate_epoch(extra_val_loader, data_mode='val', include_mask=False)
                
            self.log_metrics(train='valid')
            self.reset_metrics()

            # # save the model's weights if BinaryAUROC is higher than previous
            if self.config['save_weights']:
                if (epoch%5 == 0) or (self.config['num_epochs'] == epoch+1):
                    for j, bodypart in enumerate(self.config['anatomy_part']):
                        # save_checkpoint(state=self.models, filename=f"{self.model_checkpoints_dir}/model-{self.global_epoch}")
                        path = f"{self.model_checkpoints_dir}/{bodypart}"
                        Path(path).mkdir(parents=True, exist_ok=True)
                        print("=> Saving checkpoint")
                        torch.save(self.models[bodypart].state_dict(), f"{path}/model-{self.global_epoch}")

            self.global_epoch += 1

        # print training time
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

