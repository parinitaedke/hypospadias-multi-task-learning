import torch
import torch.nn as nn
import timm
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Vanilla_Multitask_UNET_Segmentation_Model(nn.Module):
    def __init__(self, config, testing=False):
        super(Vanilla_Multitask_UNET_Segmentation_Model, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        in_channels = config["in_channels"]
        # Down part of UNET
        for feature in config["UNET features"]:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck layer
        self.bottleneck = DoubleConv(config["UNET features"][-1], config["UNET features"][-1] * 2)

        # Up part of UNET
        for feature in reversed(config["UNET features"]):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))

        # Final convolution
        self.final_conv = nn.Conv2d(config["UNET features"][0], config["out_channels"], kernel_size=1)

        self.testing = testing
        self.dropout = nn.Dropout(p=0.5)

        # Classification heads
        # G head
        self.g_head = nn.Sequential(
            nn.LazyLinear(out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1)
        )
        # M head
        self.m_head = nn.Sequential(
            nn.LazyLinear(out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1)
        )

        # S head
        self.s_head = nn.Sequential(
            nn.LazyLinear(out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1)
        )


    def forward(self, x):
        skip_connections = []

        # print(f'Input X shape: {x.shape}')

        # Encoder: Down segments of UNET
        for i, down in enumerate(self.downs):
            x = down(x)
            # print(f'Encoder {i} conv X shape: {x.shape}')

            skip_connections.append(x)
            x = self.pool(x)
            # print(f'Encoder {i} pool X shape: {x.shape}')

        # Bottleneck
        if self.testing:
            x = self.dropout(x)

        x = self.bottleneck(x)

        print(f'Bottleneck X shape: {x.shape}')

        # Calculating the GMS classification heads
        g_score = self.g_head(x.flatten(start_dim=1))
        m_score = self.m_head(x.flatten(start_dim=1))
        s_score = self.s_head(x.flatten(start_dim=1))

        if self.testing:
            x = self.dropout(x)

        skip_connections.reverse()

        # Decoder: Up segments of UNET
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])  # resize to match sizes

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        # Classifier
        x = self.final_conv(x)

        return x, (g_score, m_score, s_score)

