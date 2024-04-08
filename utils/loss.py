# IMPORTS
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

import torch.nn.functional as F
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# CUSTOM LOSS FUNCTIONS
class SteeperMSELoss(nn.Module):
    """
    This loss function intended to further the effects of the MSE loss by multiplying the MSE loss by a coefficient,
    making the function 'steeper'.
    """
    def __init__(self, coefficient):
        super(SteeperMSELoss, self).__init__()

        self.sub_criterion = nn.MSELoss()
        self.coefficient = coefficient

    def forward(self, output, target):
        # target = torch.LongTensor(target)
        
        loss = self.sub_criterion(output, target)
        return self.coefficient * loss
    

class WeightedSoftDiceLoss(nn.Module):
    """
    This loss function gives a small weight to the background area of the label, so the background area will be 
    added to the calculation when calculating dice loss.
    
    The loss function can include the negative sample regions of the image into the loss calculation and retains the 
    advantage of the dice loss in the problem of the imbalanced distribution of the positive and negative samples.
    """
    def __init__(self, v1):
        super(WeightedSoftDiceLoss, self).__init__()
        # 0 <= v1 <= v2 <= 1; v2 = 1 - v1;
        self.v1 = v1
        self.v2 = 1 - self.v1
    def forward(self, inputs, targets, smooth=1):
        # Uncomment this if the final layer does not have an activation layer
        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.contiguous().view(-1)

        W = (targets * (self.v2 - self.v1)) + self.v1
        G_hat = W * (2 * inputs - 1)
        G = W * (2 * targets - 1)

        numerator = 2*(G_hat * G).sum() + smooth
        denominator = (G_hat**2).sum() + (G**2).sum() + smooth

        ws_dice = numerator/denominator

        return 1 - ws_dice


class DiceLoss(nn.Module):
    """
    This loss function computes the Dice score and returns the Dice loss (1 - Dice score)
    """
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def _confusion(prediction, truth):
        """ Returns the confusion matrix for the values in the `prediction` and `truth`
        tensors, i.e. the amount of positions where the values of `prediction`
        and `truth` are
        - 1 and 1 (True Positive)
        - 1 and 0 (False Positive)
        - 0 and 0 (True Negative)
        - 0 and 1 (False Negative)
        """

        confusion_vector = prediction / truth
        # Element-wise division of the 2 tensors returns a new tensor which holds a
        # unique value for each case:
        #   1     where prediction and truth are 1 (True Positive)
        #   inf   where prediction is 1 and truth is 0 (False Positive)
        #   nan   where prediction and truth are 0 (True Negative)
        #   0     where prediction is 0 and truth is 1 (False Negative)

        true_positives = torch.sum(confusion_vector == 1).item()
        false_positives = torch.sum(confusion_vector == float('inf')).item()
        true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
        false_negatives = torch.sum(confusion_vector == 0).item()

        return true_positives, false_positives, true_negatives, false_negatives

    def forward(self, inputs, targets, smooth=1):
        # Uncomment this if the final layer does not have an activation layer
        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.contiguous().view(-1)

        intersection = (inputs * targets).sum()
        numerator = (2. * intersection + smooth)
        denominator = (inputs.sum() + targets.sum() + smooth)
        dice = numerator / denominator

        # # Approach #2 -- Using the confusion matrix to calculate the dice score
        # TP, FP, TN, FN = self._confusion(inputs, targets)
        # numerator = 2 * TP + smooth
        # denominator = 2 * TP + FP + FN + smooth
        # dice = numerator / denominator
        # # print(f'Numerator: {numerator}, denominator: {denominator}')
        # dice = numerator / denominator

        return 1 - dice


class InfluenceSegmentationLoss(nn.Module):
    """
    This loss function *currently* computes the dice loss across multiple segmentation maps.
    
    A way to further this loss function would be to include some form of weighting to allow for different segmentation
    maps to have more/less influence on the score.
    """
    def __init__(self, weight_dict=None, size_average=True):
        super(InfluenceSegmentationLoss, self).__init__()
        self.weight_dict = weight_dict
        self.size_average = size_average
        
        self.sub_criterion = DiceLoss()
        

    def forward(self, targets_list, inputs_list, smooth=1):
        running_total = []
        
        for target in targets_list:
            for input in inputs_list:
                dice_loss = self.sub_criterion(input, target)
                running_total.append(1-dice_loss)
                
        
        return torch.tensor(running_total).mean()
    

class AttentionLoss(nn.Module):
    """
    This loss function is adapted from M. Rizhko's Attention Loss work.
    """
    def __init__(self, cam):
        super(AttentionLoss, self).__init__()

        self.cam = cam

    def forward(self, X, y, mask):
        # calculate predictions
        # y = y.sigmoid().data.gt(0.5)

        # create attention maps
        grayscale_cam = self.cam(X, y)
        
        # calculate loss
        # NCHW --> single channel so NHW
        n, h, w = X.shape[0], X.shape[2], X.shape[3]
        
        # (N, H, W)
        mask = torch.squeeze(mask, 1)
        a = torch.sum(mask, dim=1) # (n, w)
        b = torch.sum(a, dim=1) # (n) 

        mse_diff = (grayscale_cam - mask) ** 2
        c = torch.sum(mse_diff, dim=1)
        d = torch.sum(c, dim=1)
        
        loss = torch.sum(torch.where(b == 0, 0, d)) / n / h / w

        return loss