import torch
import torch.nn as nn


class Evaluator(nn.Module):

    def __init__(self):
        super().__init__()


        input_height = 6
        input_width = 7
        input_channels = 2


        feature_num = 4
        num_conv_layers = 3

        conv_layers = []
        in_channels = input_channels
        for i in range(num_conv_layers):
            features = (2**i) * feature_num
            conv_layers.append(nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=2, stride=1),
                nn.ReLU()
            ))
            in_channels = features

        self.conv = nn.Sequential(*conv_layers)


        self.fc = nn.Linear(in_features=(input_height-num_conv_layers)*(input_width-num_conv_layers)*features, out_features=1)

    
    def forward(self,x):

        x = self.conv(x)

        x = x.view(x.size(0),-1)

        x = self.fc(x)
        x = nn.Sigmoid()(x)

        return x
