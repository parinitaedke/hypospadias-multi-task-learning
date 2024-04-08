# IMPORTS
import torch.nn as nn
import timm

class AttentionModel(nn.Module):
    def __init__(self, config, testing=False):
        super(AttentionModel, self).__init__()
        self.config = config
        
        # set up model name
        if config['model_name'].startswith('swin'):
            model_name = f"{config['model_name']}_patch4_window7_224"
        elif config['model_name'].startswith('vit'):
            model_name = f"{config['model_name']}_patch16_224"
        elif config['model_name'].startswith('resnet'):
            model_name = config['model_name']
        else:
            raise NotImplementedError('Unknown model')

        # create model
        self.model = timm.create_model(model_name, pretrained=config['pretrained'], num_classes=1)

        # # load weights
        # if config['weights_path']:
        #     model_path = wandb.restore('model.pth', run_path=config['weights_path'])
        #     model.load_state_dict(torch.load(model_path.name))

        # freeze model
        if config['freeze']:
            submodules = [n for n, _ in self.model.named_children()]
            timm.freeze(self.model, submodules[:submodules.index(config['freeze_until']) + 1])
