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
from torchmetrics import MeanSquaredError, MeanAbsoluteError, R2Score
import torchvision

from utils.utils import save_checkpoint

from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# GLOBAL VARS
ENCODER = 'encoder'
DECODER = 'decoder'
CLASSIFICATION_HEADS = 'classification_heads'

class Trainer:
    def __init__(self, config, models, criterion, optimizers, schedulers, bodypart_lst):
        self.config = config
        self.models = models
        self.criterion = criterion
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.bodypart_lst = bodypart_lst

        self.device = config['device']
        self.global_step = 0
        self.global_epoch = 0
        self.best_auroc = 0

        self.metrics = {}
        for bodypart in self.bodypart_lst:
            self.metrics[bodypart] = MetricCollection([MeanSquaredError(),
                                                    MeanAbsoluteError(),
                                                    ])
        self.ys = []
        self.y_preds = []
        self.classification_losses = []
        
        self.train_preds = {'epoch': []}
        self.train_labels = {'epoch': []}

        self.val_preds = {'epoch': []}
        self.val_labels = {'epoch': []}

    def update_metrics(self, y_pred, y, classification_loss=[]):
        # move variables to cpu
        y_pred = torch.stack(y_pred).squeeze(2).detach().cpu()
        y = y.detach().cpu()

        # save predictions, targets and losses
        self.ys.append(y)
        self.y_preds.append(y_pred)
        
        classification_loss = [item.cpu() for item in classification_loss]
        self.classification_losses.append(classification_loss)

        # update metrics
        for i, bodypart in enumerate(self.bodypart_lst):
            self.metrics[bodypart](y_pred[i], y[i])


    def log_metrics(self, train):

        mean_bodypart_classification_losses = np.asarray(self.classification_losses).mean(axis=0)

        # log loss and calculate metrics over all batches
        for i, bodypart in enumerate(self.bodypart_lst):

            wandb.log({f"{train}_{bodypart}_classification_loss": mean_bodypart_classification_losses[i], 'epoch': self.global_epoch})

            metrics = self.metrics[bodypart].compute()

            # log metrics
            for metric in ['MeanSquaredError', 'MeanAbsoluteError']:
                wandb.log({f"{train}_{bodypart}_{metric}": metrics[metric], 'epoch': self.global_epoch})
                

    def reset_metrics(self):
        self.ys = []
        self.y_preds = []
        self.classification_losses = []
        
        for bodypart in self.bodypart_lst:
            self.metrics[bodypart].reset()
            

    def train_epoch(self, data_loader, include_mask=True, overlap_mask_penalty=False):

        # set model to train mode
        for component in [ENCODER, DECODER]: #, 'classification_heads']:
            self.models[component].train()
        
        # TODO: can just use self.bodypart_lst
        if self.config['hope_classification_heads'] and self.config['gms_classification_heads']:
            bodypart_lst = self.config['anatomy_part'] + self.config['hope_components']
        elif not self.config['hope_classification_heads'] and self.config['gms_classification_heads']:
            bodypart_lst = self.config['anatomy_part']
        elif self.config['hope_classification_heads'] and not self.config['gms_classification_heads']:
            bodypart_lst = self.config['hope_components']
        
        for subcomponent in bodypart_lst:
            self.models[CLASSIFICATION_HEADS][subcomponent].train()

        for i, batch in enumerate(tqdm(data_loader)):
            if include_mask:
                if self.config['hope_classification_heads']:
                    names, X, y, y_hope, mask = batch
                else:
                    names, X, y, mask = batch
            else:
                if self.config['hope_classification_heads']:
                    names, X, y, y_hope = batch
                else:
                    names, X, y = batch
                
            # move everything to cuda
            X = X.to(self.device)
            y = torch.stack(y).to(self.device)
            
            if self.config['hope_classification_heads']:
                y_hope = torch.stack(y_hope).to(self.device)
            
            if include_mask:
                mask = mask.to(self.device)
                
            if self.config['hope_classification_heads'] and self.config['gms_classification_heads']:
                y_lst = torch.concat([y, y_hope])
            elif not self.config['hope_classification_heads'] and self.config['gms_classification_heads']:
                y_lst = y
            elif self.config['hope_classification_heads'] and not self.config['gms_classification_heads']:
                y_lst = y_hope

            # First pass: Classification heads + encoder -------------------------------------------------------------------
            # get mask + score predictions
            classification_losses = []
            y_score_preds = []
            for j, subcomponent in enumerate(bodypart_lst):
                x, skip_connections = self.models[ENCODER](X)
                score_pred = self.models[CLASSIFICATION_HEADS][subcomponent](X)
                y_score_preds.append(score_pred)
                
                loss = self.criterion[1](score_pred.flatten(), y_lst[j].float())
                classification_losses.append(loss)
                
                self.models[CLASSIFICATION_HEADS][subcomponent].zero_grad()
                self.models[ENCODER].zero_grad()
                loss.backward() # retain_graph=True
                self.optimizers[ENCODER].step()
                self.optimizers[CLASSIFICATION_HEADS][subcomponent].step()
                self.global_step += 2

            total_classification_loss = torch.sum(torch.stack(classification_losses), dim=0)
            
            # Second pass: Segmentation + encoder ------------------------------------------------------------------------
            # get mask + score predictions
            x, skip_connections = self.models[ENCODER](X)
            y_seg_pred = self.models[DECODER](x, skip_connections)
            
            seg_losses = []
            
            # Segmentation loss ------------------------------------------------------------------------------------------
            lst_seg_pred = torch.split(y_seg_pred, 1, dim=1)  # Splits on the channel dimension
            
            if include_mask:
                # calculating the segmentation loss per mask type
                lst_gt_mask = torch.split(mask, 1, dim=1)  # Splits on the channel dimension
                
                # assert len(lst_seg_pred) == len(lst_gt_mask)
                
                for i in range(len(lst_seg_pred)):
                    seg_loss = self.criterion[0](lst_seg_pred[i], lst_gt_mask[i])
                    seg_losses.append(seg_loss)

                # calculating the segmentation loss for all masks
                total_segmentation_loss = torch.sum(torch.stack(seg_losses), dim=0)
                
                # TODO: Add foreskin overlap penalty
                if overlap_mask_penalty:
                    overlap_loss = self.config['overlap_loss_weight'] * self.criterion[2](lst_gt_mask[len(lst_seg_pred):], lst_seg_pred[:len(lst_seg_pred)])
                    
                    # total_loss += overlap_loss
                    total_segmentation_loss += overlap_loss
            
            
            # Decoder + Encoder backwards pass update
            if include_mask:
                self.models[DECODER].zero_grad()
                self.models[ENCODER].zero_grad()
                total_segmentation_loss.backward()
                self.optimizers[ENCODER].step()
                self.optimizers[DECODER].step()

                self.global_step +=2

            # log batch loss
            wandb.log({f'batch_classification_loss': total_classification_loss.item(), 'step': self.global_step})

            # save train preds
            for j, subcomponent in enumerate(bodypart_lst):
                for k, name in enumerate(names):
                    if f'{name}-{subcomponent}' not in self.train_preds:
                        self.train_preds[f'{name}-{subcomponent}'] = []
                        self.train_labels[f'{name}-{subcomponent}'] = []
                    
                    name_pred = y_score_preds[j][k]
                    name_y = y_lst[j][k]
                    
                    self.train_preds[f'{name}-{subcomponent}'].append(name_pred.item())
                    self.train_labels[f'{name}-{subcomponent}'].append(name_y.item())
            
            # save the segmentation maps per epoch to see how they evolve
            for i in range(len(lst_seg_pred)):
                batch_split_pred = torch.split(lst_seg_pred[i], 1, dim=0)  # split by batches to get msk prediction for each sample for a specific bodypart
                
                if include_mask:
                    batch_split_gt = torch.split(lst_gt_mask[i], 1, dim=0)
                
                for b_, name in enumerate(names):
                    Path(f"{self.saved_img_preds_dir}/Train/{name}/{self.config['anatomy_part'][i]}/").mkdir(parents=True, exist_ok=True)
                    torchvision.utils.save_image(batch_split_pred[b_].sigmoid(), f"{self.saved_img_preds_dir}/Train/{name}/{self.config['anatomy_part'][i]}/pred-{self.global_epoch}.png")
                    
                    if include_mask:
                        torchvision.utils.save_image(batch_split_gt[b_].float(), f"{self.saved_img_preds_dir}/Train/{name}/{self.config['anatomy_part'][i]}/gt.png")

        # log learning rate and update learning rate
        wandb.log({f'{ENCODER} learning_rate': self.optimizers[ENCODER].param_groups[0]['lr'], 'epoch': self.global_epoch})
        for component in [ENCODER, DECODER]: #, 'classification_heads']:
            self.schedulers[component].step()
        
        for subcomponent in bodypart_lst:
            self.schedulers[CLASSIFICATION_HEADS][subcomponent].step()

    def validate_epoch(self, data_loader, data_mode='val', include_mask=True):
        
        # set model to evaluation mode
        for component in [ENCODER, DECODER]: #, 'classification_heads']:
            self.models[component].eval()
        
        if self.config['hope_classification_heads'] and self.config['gms_classification_heads']:
            bodypart_lst = self.config['anatomy_part'] + self.config['hope_components']
        elif not self.config['hope_classification_heads'] and self.config['gms_classification_heads']:
            bodypart_lst = self.config['anatomy_part']
        elif self.config['hope_classification_heads'] and not self.config['gms_classification_heads']:
            bodypart_lst = self.config['hope_components']
        
        for subcomponent in bodypart_lst:
            self.models[CLASSIFICATION_HEADS][subcomponent].eval()


        for i, batch in enumerate(tqdm(data_loader)):
            
            if not include_mask:
                if self.config['hope_classification_heads']:
                    names, X, y, y_hope = batch
                else:
                    names, X, y = batch
            else:
                if self.config['hope_classification_heads']:
                    names, X, y, y_hope, mask = batch
                else:
                    names, X, y, mask = batch

            with torch.no_grad():
                # move everything to cuda
                X = X.to(self.device)
                y = torch.stack(y).to(self.device)
                
                if self.config['hope_classification_heads']:
                    y_hope = torch.stack(y_hope).to(self.device)
                
                if include_mask:
                    mask = mask.to(self.device)
                
                if self.config['hope_classification_heads'] and self.config['gms_classification_heads']:
                    y_lst = torch.concat([y, y_hope])
                elif not self.config['hope_classification_heads'] and self.config['gms_classification_heads']:
                    y_lst = y
                elif self.config['hope_classification_heads'] and not self.config['gms_classification_heads']:
                    y_lst = y_hope

                # calculate y_pred
                # get mask + score predictions
                x, skip_connections = self.models[ENCODER](X)
                y_seg_pred = self.models[DECODER](x, skip_connections)
                
                y_score_preds = []
                for j, subcomponent in enumerate(bodypart_lst):
                    score_pred = self.models[CLASSIFICATION_HEADS][subcomponent](X)
                    y_score_preds.append(score_pred)
                    

                classification_losses, seg_losses = [], []

                # calculating the segmentation loss per mask type
                lst_seg_pred = torch.split(y_seg_pred, 1, dim=1)  # splits on the channel dimension
                
                if include_mask:
                    lst_gt_mask = torch.split(mask, 1, dim=1)  # splits on the channel dimension

                    # assert len(lst_seg_pred) == len(lst_gt_mask)

                    for i in range(len(lst_seg_pred)):
                        seg_loss = self.criterion[0](lst_seg_pred[i], lst_gt_mask[i])
                        seg_losses.append(seg_loss)
                    
                # Calculating the classification loss
                for j, subcomponent in enumerate(bodypart_lst):
                    loss = self.criterion[1](y_score_preds[j].flatten(), y[j].float())
                    classification_losses.append(loss.cpu())
                    
            # save val preds
            for j, subcomponent in enumerate(bodypart_lst):
                for k, name in enumerate(names):
                    if f'{data_mode}_{name}-{subcomponent}' not in self.val_preds:
                        self.val_preds[f'{data_mode}_{name}-{subcomponent}'] = []
                        self.val_labels[f'{data_mode}_{name}-{subcomponent}'] = []
                    
                    self.val_preds[f'{data_mode}_{name}-{subcomponent}'].append(y_score_preds[j][k].item())
                    # self.val_preds[f'{data_mode}_{name}-{bodypart}'].append(y_score_preds[bodypart][k].item())
                    self.val_labels[f'{data_mode}_{name}-{subcomponent}'].append(y_lst[j][k].item())
                    
            # save the segmentation maps per epoch to see how they evolve
            for i in range(len(lst_seg_pred)):
                batch_split_pred = torch.split(lst_seg_pred[i], 1, dim=0)  # split by batches to get msk prediction for each sample for a specific bodypart
                if include_mask:
                    batch_split_gt = torch.split(lst_gt_mask[i], 1, dim=0)
                
                for b_, name in enumerate(names):
                    Path(f"{self.saved_img_preds_dir}/Val/datamode={data_mode}/{name}/{self.config['anatomy_part'][i]}/").mkdir(parents=True, exist_ok=True)
                    torchvision.utils.save_image(batch_split_pred[b_].sigmoid(), f"{self.saved_img_preds_dir}/Val/datamode={data_mode}/{name}/{self.config['anatomy_part'][i]}/pred-{self.global_epoch}.png")
                    if include_mask:
                        torchvision.utils.save_image(batch_split_gt[b_].float(), f"{self.saved_img_preds_dir}/Val/datamode={data_mode}/{name}/{self.config['anatomy_part'][i]}/gt.png")    
                

            # add batch predictions, ground truth and loss to metrics
            self.update_metrics(y_score_preds, y_lst, classification_loss=classification_losses)
            

    def run(self, train_loader, valid_loader, extra_train_ds_loaders, extra_val_ds_loaders):
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
            
            self.train_preds['epoch'].append(self.global_epoch)
            self.train_labels['epoch'].append(self.global_epoch)

            # train epoch
            self.train_epoch(train_loader, overlap_mask_penalty=any(self.config['overlap_bodyparts']))
            
            for extra_train_loader in extra_train_ds_loaders:
                self.train_epoch(extra_train_loader, include_mask=False)

            # validate and log on train data
            self.val_preds['epoch'].append(self.global_epoch)
            self.val_labels['epoch'].append(self.global_epoch)

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
                    # save_checkpoint(state=self.models, filename=f"{self.model_checkpoints_dir}/model-{self.global_epoch}")
                    model_save_path = f"{self.model_checkpoints_dir}/model-{self.global_epoch}"
                    Path(model_save_path).mkdir(parents=True, exist_ok=True)
                    
                    print("=> Saving checkpoint")
                    for component in ['encoder', 'decoder']:
                        torch.save(self.models[component].state_dict(), f"{model_save_path}/{component}")

                    for subcomponent in self.bodypart_lst:
                        torch.save(self.models['classification_heads'][subcomponent].state_dict(), f"{model_save_path}/classification_heads-{subcomponent}")
            self.global_epoch += 1

        train_preds_df = pd.DataFrame.from_dict(self.train_preds)
        train_labels_df = pd.DataFrame.from_dict(self.train_labels)
        val_preds_df = pd.DataFrame.from_dict(self.val_preds)
        val_labels_df = pd.DataFrame.from_dict(self.val_labels)

        train_preds_df.to_csv(f"{self.saved_preds_dir}/train_preds.csv", index=False)
        train_labels_df.to_csv(f"{self.saved_preds_dir}/train_labels.csv", index=False)
        val_preds_df.to_csv(f"{self.saved_preds_dir}/val_preds.csv", index=False)
        val_labels_df.to_csv(f"{self.saved_preds_dir}/val_labels.csv", index=False)

        wandb.log({"Train predictions": wandb.Table(data=train_preds_df)})
        wandb.log({"Train labels": wandb.Table(data=train_labels_df)})
        wandb.log({"Val predictions": wandb.Table(data=val_preds_df)})
        wandb.log({"Val labels": wandb.Table(data=val_labels_df)})
        
        # print training time
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

