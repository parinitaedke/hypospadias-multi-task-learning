import torch
import torch.nn as nn
import timm


class Vanilla_Multiclass_Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        if config['model_name'].startswith('vit'):
            model_name = f"{config['model_name']}_patch16_224"
        elif config['model_name'].startswith('resnet'):
            model_name = config['model_name']
        else:
            raise NotImplementedError('Unknown model')

        # create model for sharing
        self.model = timm.create_model(model_name, pretrained=config['pretrained'])

        # freeze model
        if config['freeze']:
            submodules = [n for n, _ in self.model.named_children()]
            timm.freeze(self.model, submodules[:submodules.index(config['freeze_until']) + 1])

        # G head
        self.g_head = nn.Sequential(
            nn.Linear(in_features=1000, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1)
        )
        # M head
        self.m_head = nn.Sequential(
            nn.Linear(in_features=1000, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1)
        )

        # S head
        self.s_head = nn.Sequential(
            nn.Linear(in_features=1000, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1)
        )

    def forward(self, x):
        interim_output = self.model(x)

        g_score = self.g_head(interim_output)
        m_score = self.m_head(interim_output)
        s_score = self.s_head(interim_output)

        return g_score, m_score, s_score
