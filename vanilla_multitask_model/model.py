import torch
import torch.nn as nn
import timm


class Vanilla_Multitask_Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config

        if self.config['model_name'].startswith('vit'):
            model_name = f"{self.config['model_name']}_patch16_224"
        elif self.config['model_name'].startswith('resnet'):
            model_name = self.config['model_name']
        else:
            raise NotImplementedError('Unknown model')

        # create model for sharing
        self.model = timm.create_model(model_name, pretrained=self.config['pretrained'])

        # freeze model
        if self.config['freeze']:
            submodules = [n for n, _ in self.model.named_children()]
            timm.freeze(self.model, submodules[:submodules.index(self.config['freeze_until']) + 1])

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
        
        if self.config['hope_classfication_heads']:
            
            # Meatus position head
            self.meatus_pos_head = nn.Sequential(
                nn.Linear(in_features=1000, out_features=512),
                nn.ReLU(),
                nn.Linear(in_features=512, out_features=256),
                nn.ReLU(),
                nn.Linear(in_features=256, out_features=64),
                nn.ReLU(),
                nn.Linear(in_features=64, out_features=1)
            )
            
            # Meatus shape head
            self.meatus_shape_head = nn.Sequential(
                nn.Linear(in_features=1000, out_features=512),
                nn.ReLU(),
                nn.Linear(in_features=512, out_features=256),
                nn.ReLU(),
                nn.Linear(in_features=256, out_features=64),
                nn.ReLU(),
                nn.Linear(in_features=64, out_features=1)
            )
            
            # Glans shape head
            self.glans_shape_head = nn.Sequential(
                nn.Linear(in_features=1000, out_features=512),
                nn.ReLU(),
                nn.Linear(in_features=512, out_features=256),
                nn.ReLU(),
                nn.Linear(in_features=256, out_features=64),
                nn.ReLU(),
                nn.Linear(in_features=64, out_features=1)
            )
            
            # Penile skin shape head
            self.penile_skin_shape_head = nn.Sequential(
                nn.Linear(in_features=1000, out_features=512),
                nn.ReLU(),
                nn.Linear(in_features=512, out_features=256),
                nn.ReLU(),
                nn.Linear(in_features=256, out_features=64),
                nn.ReLU(),
                nn.Linear(in_features=64, out_features=1)
            )
            
            # Torsion head
            self.torsion_head = nn.Sequential(
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
        
        if self.config['hope_classfication_heads']:
            meatus_pos_score = self.meatus_pos_head(interim_output)
            meatus_shape_score = self.meatus_shape_head(interim_output)
            glans_shape_score = self.glans_shape_head(interim_output)
            penile_skin_shape_score = self.penile_skin_shape_head(interim_output)
            torsion_score = self.torsion_head(interim_output)
            

        if self.config['hope_classifcation_heads']:
            return (g_score, m_score, s_score), (meatus_pos_score, meatus_shape_score, glans_shape_score, penile_skin_shape_score, torsion_score)
        else:
            return g_score, m_score, s_score
