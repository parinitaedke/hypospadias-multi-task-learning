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
        # inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        TP, FP, TN, FN = self._confusion(inputs, targets)
        numerator = 2 * TP + smooth
        denominator = 2 * TP + FP + FN + smooth
        dice = numerator / denominator

        # print(f'Numerator: {numerator}, denominator: {denominator}')

        dice = numerator / denominator
        return 1 - dice


class InfluenceSegmentationLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(InfluenceSegmentationLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # TODO: Implement forward call
        pass