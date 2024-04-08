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

from utils.utils import save_checkpoint

from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path


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
        # if self.config['gms_classification_heads']:
        #     for bodypart in self.config['anatomy_part']:
        #         self.metrics[bodypart] = MetricCollection([MeanSquaredError(),
        #                                                 MeanAbsoluteError(),
        #                                                 # R2Score()
        #                                                 ])
        
        # if self.config['hope_classification_heads']:
        #     for hope_component in (['Position of meatus', 'Shape of meatus', 'Shape of glans', 'Shape of skin', 'Torsion']):
        #         self.metrics[hope_component] = MetricCollection([MeanSquaredError(),
        #                                                         MeanAbsoluteError(),
        #                                                         ])
        
        self.losses = []
        self.ys = []
        self.y_preds = []
        # if self.config['gms_classification_heads']:                                                        
        #     self.ys = []
        #     self.y_preds = []
            
        # if self.config['hope_classification_heads']:
        #     self.ys_hope = []
        #     self.y_hope_preds = []

        self.train_preds = {'epoch': []}
        self.train_labels = {'epoch': []}

        self.val_preds = {'epoch': []}
        self.val_labels = {'epoch': []}

    def update_metrics(self, y_pred=None, y=None, loss=[]):
        # move variables to cpu
        y_pred = torch.stack(y_pred).squeeze(2).detach().cpu()
        y = y.detach().cpu()

        # save predictions, targets and losses
        self.ys.append(y)
        self.y_preds.append(y_pred)
        
        # if self.config['gms_classification_heads']:
        #     # move variables to cpu
        #     y_pred = torch.stack(y_pred).squeeze(2).detach().cpu()
        #     y = y.detach().cpu()

        #     # save predictions, targets and losses
        #     self.ys.append(y)
        #     self.y_preds.append(y_pred)
        
        # if self.config['hope_classification_heads']:
        #     y_hope_pred = torch.stack(y_hope_pred).squeeze(2).detach().cpu()
        #     y_hope = y_hope.detach().cpu()
            
        #     self.ys_hope.append(y)
        #     self.y_hope_preds.append(y_pred)
        
        self.losses.append(loss)
        # update metrics
        for i, bodypart in enumerate(self.bodypart_lst):
            self.metrics[bodypart](y_pred[i], y[i])
            
        # if self.config['gms_classification_heads']:
        #     # update metrics
        #     for i, bodypart in enumerate(self.config['anatomy_part']):
        #         self.metrics[bodypart](y_pred[i], y[i])
        
        # if self.config['hope_classification_heads']:
        #     for k, hope_component in enumerate(['Position of meatus', 'Shape of meatus', 'Shape of glans', 'Shape of skin', 'Torsion']):
        #         self.metrics[hope_component](y_hope_pred[k], y_hope[k])

    def log_metrics(self, train):
        losses = []
        for l in self.losses:
            losses.append([item.cpu() if torch.is_tensor(item) else item for item in l])
        self.losses = losses
        mean_bodypart_losses = np.asarray(self.losses).mean(axis=0)

        # log loss and calculate metrics over all batches
        for i, bodypart in enumerate(self.bodypart_lst):

            wandb.log({f"{train}_{bodypart}_loss": mean_bodypart_losses[i], 'epoch': self.global_epoch})

            metrics = self.metrics[bodypart].compute()

            # log metrics
            for metric in ['MeanSquaredError', 'MeanAbsoluteError']:
                wandb.log({f"{train}_{bodypart}_{metric}": metrics[metric], 'epoch': self.global_epoch})
                
                
        # # log loss and calculate metrics over all batches
        # i, k = 0, 0
        # if self.config['gms_classification_heads']:
        #     for i, bodypart in enumerate(self.config['anatomy_part']):

        #         wandb.log({f"{train}_{bodypart}_loss": mean_bodypart_losses[i], 'epoch': self.global_epoch})

        #         metrics = self.metrics[bodypart].compute()

        #         # log metrics
        #         for metric in ['MeanSquaredError', 'MeanAbsoluteError']:
        #             wandb.log({f"{train}_{bodypart}_{metric}": metrics[metric], 'epoch': self.global_epoch})
        
        # if self.config['hope_classification_heads']:
        #     for k, hope_component in enumerate(['Position of meatus', 'Shape of meatus', 'Shape of glans', 'Shape of skin', 'Torsion']):
        #         wandb.log({f"{train}_{hope_component}_classification_loss": mean_bodypart_losses[i+k], 'epoch': self.global_epoch})
                
        #         metrics = self.metrics[hope_component].compute()
                
        #         # log metrics
        #         for metric in ['MeanSquaredError', 'MeanAbsoluteError']:
        #             wandb.log({f"{train}_{hope_component}_{metric}": metrics[metric], 'epoch': self.global_epoch})
                

    def reset_metrics(self):
        self.losses = []
        
        self.ys = []
        self.y_preds = []
        
        for bodypart in self.bodypart_lst:
            self.metrics[bodypart].reset()
            
        # if self.config['gms_classification_heads']:
        #     self.ys = []
        #     self.y_preds = []
        #     for bodypart in self.config['anatomy_part']:
        #         self.metrics[bodypart].reset()
        
        # if self.config['hope_classification_heads']:
        #     self.ys_hope = []
        #     self.y_hope_preds = []
            
        #     for hope_component in (['Position of meatus', 'Shape of meatus', 'Shape of glans', 'Shape of skin', 'Torsion']):
        #         self.metrics[hope_component].reset()

    def train_epoch(self, data_loader):

        # set model to train mode
        # self.models.train()
        self.models['encoder'].train()
        
        for subcomponent in self.bodypart_lst:
            self.models['classification_heads'][subcomponent].train()
        
        # TODO: EDIT FROM HERE ON NOV 1ST MORNING   

        for i, batch in enumerate(tqdm(data_loader)):
            
            if self.config['gms_classification_heads'] and not self.config['hope_classification_heads']:
                names, X, y = batch[0], batch[1], batch[2]
            elif not self.config['gms_classification_heads'] and self.config['hope_classification_heads']:
                names, X, y_hope = batch[0], batch[1], batch[2]
            else:
                names, X, y, y_hope = batch[0], batch[1], batch[2], batch[3]
            
            # move everything to cuda
            X = X.to(self.device)
            
            if self.config['gms_classification_heads']:
                y = torch.stack(y).to(self.device)
            
            if self.config['hope_classification_heads']:
                y_hope = torch.stack(y_hope).to(self.device)
            
            if self.config['hope_classification_heads'] and self.config['gms_classification_heads']:
                y_lst = torch.concat([y, y_hope])
            elif not self.config['hope_classification_heads'] and self.config['gms_classification_heads']:
                y_lst = y
            elif self.config['hope_classification_heads'] and not self.config['gms_classification_heads']:
                y_lst = y_hope

            # model prediction
            losses = []
            y_score_preds = []
            for j, subcomponent in enumerate(self.bodypart_lst):
                x = self.models['encoder'](X)
                score_pred = self.models['classification_heads'][subcomponent](x)
                y_score_preds.append(score_pred)
                
                loss = self.criterion(score_pred.flatten(), y_lst[j].float())
                losses.append(loss)
                
                self.models['classification_heads'][subcomponent].zero_grad()
                self.models['encoder'].zero_grad()
                loss.backward() # retain_graph=True
                self.optimizers['encoder'].step()
                self.optimizers['classification_heads'][subcomponent].step()
                self.global_step += 2

            total_loss = torch.sum(torch.stack(losses), dim=0)
            
            # if self.config['gms_classification_heads'] and not self.config['hope_classification_heads']:
            #     y_preds = self.models(X)
            # elif not self.config['gms_classification_heads'] and self.config['hope_classification_heads']:
            #     y_hope_preds = self.models(X)
            # else:
            #     y_preds, y_hope_preds = self.models(X)

            # losses = []
            
            # # GMS
            # if self.config['gms_classification_heads']:
            #     for j, bodypart in enumerate(self.config['anatomy_part']):
            #         loss = self.criterion(y_preds[j].flatten(), y[j].float())
            #         losses.append(loss)
            
            # # HOPE
            # if self.config['hope_classification_heads']:
            #     for k, hope_component in enumerate(['Position of meatus', 'Shape of meatus', 'Shape of glans', 'Shape of skin', 'Torsion']):
            #         loss = self.criterion(y_hope_preds[k].flatten(), y_hope[k].float())
            #         losses.append(loss)

            # # calculate loss and optimize model
            # total_loss = torch.sum(torch.stack(losses), dim=0)
            # self.models.zero_grad()
            # total_loss.backward()
            # self.optimizers.step()

            # log batch loss
            wandb.log({f'batch_loss': total_loss.item(), 'step': self.global_step})

            # save train preds
            
            # save train preds
            for j, subcomponent in enumerate(self.bodypart_lst):
                for k, name in enumerate(names):
                    if f'{name}-{subcomponent}' not in self.train_preds:
                        self.train_preds[f'{name}-{subcomponent}'] = []
                        self.train_labels[f'{name}-{subcomponent}'] = []
                    
                    name_pred = y_score_preds[j][k]
                    name_y = y_lst[j][k]
                    
                    self.train_preds[f'{name}-{subcomponent}'].append(name_pred.item())
                    self.train_labels[f'{name}-{subcomponent}'].append(name_y.item())
            
            # # GMS
            # if self.config['gms_classification_heads']:
            #     for j, bodypart in enumerate(self.config['anatomy_part']):
            #         for k, name in enumerate(names):
            #             if f'{name}-{bodypart}' not in self.train_preds:
            #                 self.train_preds[f'{name}-{bodypart}'] = []
            #                 self.train_labels[f'{name}-{bodypart}'] = []

            #             name_pred = y_preds[j][k]
            #             name_y = y[j][k]
            #             self.train_preds[f'{name}-{bodypart}'].append(name_pred.item())
            #             self.train_labels[f'{name}-{bodypart}'].append(name_y.item())

            # # HOPE
            # if self.config['hope_classification_heads']:
            #     for m, hope_component in enumerate(['Position of meatus', 'Shape of meatus', 'Shape of glans', 'Shape of skin', 'Torsion']):
            #         for n, name in enumerate(names):
            #             if f'{name}-{hope_component}' not in self.train_preds:
            #                 self.train_preds[f'{name}-{hope_component}'] = []
            #                 self.train_labels[f'{name}-{hope_component}'] = []

            #             name_pred = y_hope_preds[m][n]
            #             name_y = y_hope[m][n]
            #             self.train_preds[f'{name}-{hope_component}'].append(name_pred.item())
            #             self.train_labels[f'{name}-{hope_component}'].append(name_y.item())

            
            self.global_step += 1

        # log learning rate and update learning rate
        wandb.log({f'encoder learning_rate': self.optimizers['encoder'].param_groups[0]['lr'], 'epoch': self.global_epoch})
        self.schedulers['encoder'].step()
        for subcomponent in self.bodypart_lst:
            self.schedulers['classification_heads'][subcomponent].step()

    def validate_epoch(self, data_loader, data_mode='val'):
        # TODO : EDIT

        # set model to evaluation mode
        self.models['encoder'].eval()
        
        for subcomponent in self.bodypart_lst:
            self.models['classification_heads'][subcomponent].eval()

        for i, batch in enumerate(tqdm(data_loader)):
            y_preds = []
            losses = []

            if self.config['gms_classification_heads'] and not self.config['hope_classification_heads']:
                names, X, y = batch[0], batch[1], batch[2]
            elif not self.config['gms_classification_heads'] and self.config['hope_classification_heads']:
                names, X, y_hope = batch[0], batch[1], batch[2]
            else:
                names, X, y, y_hope = batch[0], batch[1], batch[2], batch[3]
            
            with torch.no_grad():
                # move everything to cuda
                X = X.to(self.device)
                
                if self.config['gms_classification_heads']:
                    y = torch.stack(y).to(self.device)
                
                if self.config['hope_classification_heads']:
                    y_hope = torch.stack(y_hope).to(self.device)
                
                if self.config['hope_classification_heads'] and self.config['gms_classification_heads']:
                    y_lst = torch.concat([y, y_hope])
                elif not self.config['hope_classification_heads'] and self.config['gms_classification_heads']:
                    y_lst = y
                elif self.config['hope_classification_heads'] and not self.config['gms_classification_heads']:
                    y_lst = y_hope

                # calculate y_pred
                # model prediction
                x = self.models['encoder'](X)
                y_score_preds = []
                for j, subcomponent in enumerate(self.bodypart_lst):
                    score_pred = self.models['classification_heads'][subcomponent](x)
                    y_score_preds.append(score_pred)
                
                # if self.config['gms_classification_heads'] and not self.config['hope_classification_heads']:
                #     y_preds = self.models(X)
                # elif not self.config['gms_classification_heads'] and self.config['hope_classification_heads']:
                #     y_hope_preds = self.models(X)
                # else:
                #     y_preds, y_hope_preds = self.models(X)

                losses = []
                for j, subcomponent in enumerate(self.bodypart_lst):
                    loss = self.criterion(y_score_preds[j].flatten(), y_lst[j].float())
                    losses.append(loss.cpu())
                    
                # if self.config['gms_classification_heads']:
                #     for j, bodypart in enumerate(self.config['anatomy_part']):
                #         loss = self.criterion(y_preds[j].flatten(), y[j].float())
                #         losses.append(loss.item())
                
                # if self.config['hope_classification_heads']:
                #     for k, hope_component in enumerate(['Position of meatus', 'Shape of meatus', 'Shape of glans', 'Shape of skin', 'Torsion']):
                #         loss = self.criterion(y_hope_preds[k].flatten(), y_hope[k].float())
                #         losses.append(loss)
                # save val preds
                for j, subcomponent in enumerate(self.bodypart_lst):
                    for k, name in enumerate(names):
                        if f'{data_mode}_{name}-{subcomponent}' not in self.val_preds:
                            self.val_preds[f'{data_mode}_{name}-{subcomponent}'] = []
                            self.val_labels[f'{data_mode}_{name}-{subcomponent}'] = []
                        
                        self.val_preds[f'{data_mode}_{name}-{subcomponent}'].append(y_score_preds[j][k].item())
                        # self.val_preds[f'{data_mode}_{name}-{bodypart}'].append(y_score_preds[bodypart][k].item())
                        self.val_labels[f'{data_mode}_{name}-{subcomponent}'].append(y_lst[j][k].item())
                    
                # # save val preds
                # if self.config['gms_classification_heads']:
                #     for j, bodypart in enumerate(self.config['anatomy_part']):
                #         for k, name in enumerate(names):
                #             if f'{data_mode}_{name}-{bodypart}' not in self.val_preds:
                #                 self.val_preds[f'{data_mode}_{name}-{bodypart}'] = []
                #                 self.val_labels[f'{data_mode}_{name}-{bodypart}'] = []

                #             self.val_preds[f'{data_mode}_{name}-{bodypart}'].append(y_preds[j][k].item())
                #             self.val_labels[f'{data_mode}_{name}-{bodypart}'].append(y[j][k].item())
                # # HOPE
                # if self.config['hope_classification_heads']:
                #     for m, hope_component in enumerate(['Position of meatus', 'Shape of meatus', 'Shape of glans', 'Shape of skin', 'Torsion']):
                #         for n, name in enumerate(names):
                #             if f'{data_mode}_{name}-{hope_component}' not in self.val_preds:
                #                 self.val_preds[f'{data_mode}_{name}-{hope_component}'] = []
                #                 self.val_labels[f'{data_mode}_{name}-{hope_component}'] = []

                #             self.val_preds[f'{data_mode}_{name}-{hope_component}'].append(y_hope_preds[m][n].item())
                #             self.val_labels[f'{data_mode}_{name}-{hope_component}'].append(y_hope[m][n].item())
                
                # add batch predictions, ground truth and loss to metrics
                self.update_metrics(y_score_preds, y_lst, loss=losses)

                # # add batch predictions, ground truth and loss to metrics
                # if self.config['gms_classification_heads'] and not self.config['hope_classification_heads']:
                #     self.update_metrics(y_pred=y_preds, y=y, loss=losses)
                # elif not self.config['gms_classification_heads'] and self.config['hope_classification_heads']:
                #     self.update_metrics(y_hope_pred=y_hope_preds, y_hope=y_hope, loss=losses)
                # else:
                #     self.update_metrics(y_pred=y_preds, y=y, y_hope_pred=y_hope_preds, y_hope=y_hope, loss=losses)
            

    def run(self, train_loader, valid_loader, extra_train_ds_loaders, extra_val_ds_loaders):
        # TODO : EDIT
        since = time.time()

        # Make saved preds and checkpoints directory
        self.saved_preds_dir = f'{wandb.run.dir}/saved_preds'
        self.model_checkpoints_dir = f'{wandb.run.dir}/model_checkpoints'
        
        Path(self.saved_preds_dir).mkdir(parents=True, exist_ok=True)
        Path(self.model_checkpoints_dir).mkdir(parents=True, exist_ok=True)

        for epoch in range(self.config['num_epochs']):
            print('Epoch:', epoch)
            
            self.train_preds['epoch'].append(self.global_epoch)
            self.train_labels['epoch'].append(self.global_epoch)

            # train epoch
            self.train_epoch(train_loader)
            
            for extra_train_loader in extra_train_ds_loaders:
                self.train_epoch(extra_train_loader)

            # validate and log on train data
            self.val_preds['epoch'].append(self.global_epoch)
            self.val_labels['epoch'].append(self.global_epoch)

            # validate and log on train data
            self.validate_epoch(train_loader, data_mode='train')
            
            for extra_train_loader in extra_train_ds_loaders:
                self.validate_epoch(extra_train_loader, data_mode='train')

            self.log_metrics(train='train')
            self.reset_metrics()

            # validate and log on valid data
            self.validate_epoch(valid_loader, data_mode='val')
            
            for extra_val_loader in extra_val_ds_loaders:
                self.validate_epoch(extra_val_loader, data_mode='val')
                
            self.log_metrics(train='valid')
            self.reset_metrics()

            # # save the model's weights if BinaryAUROC is higher than previous
            if self.config['save_weights']:
                if (epoch%5 == 0) or (self.config['num_epochs'] == epoch+1):
                    model_save_path = f"{self.model_checkpoints_dir}/model-{self.global_epoch}"
                    Path(model_save_path).mkdir(parents=True, exist_ok=True)
                    
                    print("=> Saving checkpoint")
                    torch.save(self.models['encoder'].state_dict(), f"{model_save_path}/{'encoder'}")

                    for subcomponent in self.bodypart_lst:
                        torch.save(self.models['classification_heads'][subcomponent].state_dict(), f"{model_save_path}/classification_heads-{subcomponent}")
                    # save_checkpoint(state=self.models, filename=f"{self.model_checkpoints_dir}/model-{self.global_epoch}")

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

