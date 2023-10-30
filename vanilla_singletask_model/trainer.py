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
            self.metrics[bodypart] = MetricCollection([MeanSquaredError(), 
                                         MeanAbsoluteError(),
                                         #R2Score()
                                        ])
        self.ys = []
        self.y_preds = []
        self.losses = []

        self.train_preds = {'epoch': []}
        self.train_labels = {'epoch': []}

        self.val_preds = {'epoch': []}
        self.val_labels = {'epoch': []}

    def update_metrics(self, y_pred, y, loss):
        # move variables to cpu
        y_pred = torch.stack(y_pred).squeeze(2).detach().cpu()
        y = y.detach().cpu()

        # save predictions, targets and losses
        self.ys.append(y)
        self.y_preds.append(y_pred)
        self.losses.append(loss)

        # update metrics
        for i, bodypart in enumerate(self.config['anatomy_part']):
            self.metrics[bodypart](y_pred[i], y[i])

    def log_metrics(self, train):

        mean_bodypart_losses = np.asarray(self.losses).mean(axis=0)

        # log loss and calculate metrics over all batches 
        for i, bodypart in enumerate(self.config['anatomy_part']):
            
            wandb.log({f"{train}_{bodypart}_loss": mean_bodypart_losses[i], 'epoch': self.global_epoch})

            metrics = self.metrics[bodypart].compute()

            # log metrics
            for metric in ['MeanSquaredError', 'MeanAbsoluteError']:
                wandb.log({f"{train}_{bodypart}_{metric}": metrics[metric], 'epoch': self.global_epoch})


    def reset_metrics(self):
        self.ys = []
        self.y_preds = []
        self.losses = []
        for bodypart in self.config['anatomy_part']:
            self.metrics[bodypart].reset()

    def train_epoch(self, data_loader):
        # set model to train mode
        for bodypart in self.config['anatomy_part']:
            self.models[bodypart].train()

        for i, batch in enumerate(tqdm(data_loader)):
            
            names, X, y = batch[0], batch[1], batch[2]
            
            # move everything to cuda
            X = X.to(self.device)
            y = torch.stack(y).to(self.device)
            # mask = mask.to(self.device)

            for j, bodypart in enumerate(self.config['anatomy_part']):
                y_pred = self.models[bodypart](X)

                loss = self.criterion(torch.flatten(y_pred), y[j].float())

                # calculate loss and optimize model
                self.models[bodypart].zero_grad()
                loss.backward()
                self.optimizers[bodypart].step()

                # log batch loss
                wandb.log({f'{bodypart}_batch_loss': loss.item(), 'step': self.global_step})

                # save train preds
                for k, name in enumerate(names):
                    if f'{name}-{bodypart}' not in self.train_preds:
                        self.train_preds[f'{name}-{bodypart}'] = []
                        self.train_labels[f'{name}-{bodypart}'] = []

                    self.train_preds[f'{name}-{bodypart}'].append(y_pred[k].item())
                    self.train_labels[f'{name}-{bodypart}'].append(y[j][k].item())

            self.global_step += 1

        # log learning rate and update learning rate
        for bodypart in self.config['anatomy_part']:
            wandb.log({f'{bodypart}_learning_rate': self.optimizers[bodypart].param_groups[0]['lr'], 'epoch': self.global_epoch})
            self.schedulers[bodypart].step()

    def validate_epoch(self, data_loader, data_mode='val'):
        # TODO : EDIT

        # set model to evaluation mode
        for bodypart in self.config['anatomy_part']:
            self.models[bodypart].eval()

        for i, batch in enumerate(tqdm(data_loader)):
            y_preds = []
            losses = []

            names, X, y = batch[0], batch[1], batch[2]

            with torch.no_grad():
                # move everything to cuda
                X = X.to(self.device)
                y = torch.stack(y).to(self.device)
                # mask = mask.to(self.device)

                for j, bodypart in enumerate(self.config['anatomy_part']):
                    # calculate y_pred
                    # y_pred = self.model(X).reshape(-1)
                    y_pred = self.models[bodypart](X)


                    loss = self.criterion(torch.flatten(y_pred), y[j].float())
                    y_preds.append(y_pred)
                    losses.append(loss.item())

                    # save val preds
                    for k, name in enumerate(names):
                        if f'{data_mode}_{name}-{bodypart}' not in self.val_preds:
                            self.val_preds[f'{data_mode}_{name}-{bodypart}'] = []
                            self.val_labels[f'{data_mode}_{name}-{bodypart}'] = []

                        self.val_preds[f'{data_mode}_{name}-{bodypart}'].append(y_pred[k].item())
                        self.val_labels[f'{data_mode}_{name}-{bodypart}'].append(y[j][k].item())

           
            # add batch predictions, ground truth and loss to metrics
            self.update_metrics(y_preds, y, losses)


    def run(self, train_loader, valid_loader, extra_train_ds_loaders, extra_val_ds_loaders):
        # TODO : EDIT
        since = time.time()

        saved_preds_dir = f'{wandb.run.dir}/saved_preds'
        model_checkpoints_dir = f'{wandb.run.dir}/model_checkpoints'

        Path(saved_preds_dir).mkdir(parents=True, exist_ok=True)
        for bodypart in self.config['anatomy_part']:
            Path(f'{model_checkpoints_dir}/{bodypart}').mkdir(parents=True, exist_ok=True)

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
                    for bodypart in self.config['anatomy_part']:
                        save_checkpoint(state=self.models[bodypart], filename= f"{model_checkpoints_dir}/{bodypart}/model-{self.global_epoch}")

            self.global_epoch += 1

        train_preds_df = pd.DataFrame.from_dict(self.train_preds)
        train_labels_df = pd.DataFrame.from_dict(self.train_labels)
        val_preds_df = pd.DataFrame.from_dict(self.val_preds)
        val_labels_df = pd.DataFrame.from_dict(self.val_labels)

        train_preds_df.to_csv(f"{saved_preds_dir}/train_preds.csv", index=False)
        train_labels_df.to_csv(f"{saved_preds_dir}/train_labels.csv", index=False)
        val_preds_df.to_csv(f"{saved_preds_dir}/val_preds.csv", index=False)
        val_labels_df.to_csv(f"{saved_preds_dir}/val_labels.csv", index=False)

        wandb.log({"Train predictions": wandb.Table(data=train_preds_df)})
        wandb.log({"Train labels": wandb.Table(data=train_labels_df)})
        wandb.log({"Val predictions": wandb.Table(data=val_preds_df)})
        wandb.log({"Val labels": wandb.Table(data=val_labels_df)})
        # print training time
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    