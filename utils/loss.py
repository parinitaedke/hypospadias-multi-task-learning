import torch
import torch.nn as nn
import numpy as np
import pandas as pd

import torch.nn.functional as F

class SteeperMSELoss(nn.Module):
    def __init__(self, coefficient):
        super(SteeperMSELoss, self).__init__()

        self.sub_criterion = nn.MSELoss()
        self.coefficient = coefficient

    def forward(self, output, target):
        # target = torch.LongTensor(target)
        
        loss = self.sub_criterion(output, target)
   
        return self.coefficient * loss
    
class WeightedSoftDiceLoss(nn.Module):
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
    def __init__(self, weight_dict=None, size_average=True):
        super(InfluenceSegmentationLoss, self).__init__()
        self.weight_dict = weight_dict
        self.size_average = size_average
        
        self.sub_criterion = DiceLoss()
        

    def forward(self, targets_list, inputs_list, smooth=1):
        # TODO: Implement forward call
        running_total = []
        
        for target in targets_list:
            for input in inputs_list:
                dice_loss = self.sub_criterion(input, target)
                running_total.append(1-dice_loss)
                
        
        return torch.tensor(running_total).mean()