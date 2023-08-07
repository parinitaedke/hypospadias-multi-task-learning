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

    def forward(self, inputs, targets, smooth=1):
        # inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = np.logical_not(np.logical_xor(inputs, targets)).sum()

        numerator = 2 * intersection + smooth
        denominator = torch.numel(inputs) + torch.numel(targets) + smooth

        # print(f'Numerator: {numerator}, denominator: {denominator}')

        dice = numerator / denominator
        return 1 - dice


class InfluenceSegmentationLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(InfluenceSegmentationLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # TODO: Implement forward call
        pass