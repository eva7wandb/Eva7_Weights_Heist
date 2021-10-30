import torch
import torch.nn as nn
import torch.nn.functional as F


class conv_block(nn.Module):
    def __init__(
        self,
        in_channels, out_channels,
        kernel_size=(3, 3), padding=0,
        bias=False,
        activation=True,
        dropout_value=0.01,
        batch_norm=False,
        layer_norm=False,
        group_norm=False,
        group_norm_groups=2
    ):
        super().__init__()
        conv_layer = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, 
            kernel_size=kernel_size, padding=padding, 
            bias=bias
        )
        elements = [conv_layer]
        
        if activation:
            elements.append(nn.ReLU())
            
        ## regularizers
        if batch_norm:
            elements.append(nn.BatchNorm2d(out_channels))
        
        if layer_norm:
            elements.append(nn.GroupNorm(1, out_channels))
        
        if group_norm:
            elements.append(nn.GroupNorm(group_norm_groups, out_channels))
        
        if dropout_value:
            elements.append(nn.Dropout(dropout_value))
        
        self.layer = nn.Sequential(*elements)
    
    def forward(self, x):
        x = self.layer(x)
        return x
        

class Net(nn.Module):
    def __init__(
        self,
        dropout_value=0.01, ## Note dropout_value = 0, is same as no dropout
        batch_norm=True,
        layer_norm=False,
        group_norm=False,
        group_norm_groups=2,
    ):
        super(Net, self).__init__()
        
        regularization_config = {
            'dropout_value': dropout_value,
            'batch_norm': batch_norm,
            'layer_norm': layer_norm,
            'group_norm': group_norm,
            'group_norm_groups': group_norm_groups,
        }
        regularization_off = {
            'activation': False,
            'dropout_value': 0,
            'batch_norm': False,
            'layer_norm': False,
            'group_norm': False,
        }
        
        # INPUT BLOCK
        self.convblock1 = conv_block(
            in_channels=1, out_channels=8, **regularization_config,
        )
        
        # CONVOLUTION BLOCK 1
        self.convblock2 = conv_block(
            in_channels=8, out_channels=16, **regularization_config,
        )
        # TRANSITION BLOCK 1
        self.convblock3 = conv_block(
            in_channels=16, out_channels=8, **regularization_off,
            kernel_size=(1, 1),
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # CONVOLUTION BLOCK 2
        self.convblock4 = conv_block(
            in_channels=8, out_channels=12, **regularization_config,
        )
        self.convblock6 = conv_block(
            in_channels=12, out_channels=14, **regularization_config,
        )
        self.convblock7 = conv_block(
            in_channels=14, out_channels=14, **regularization_config,
            padding=1
        )
        
        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        )
        self.convblock8 = conv_block(
            in_channels=14, out_channels=10, **regularization_off,
            kernel_size=(1, 1)
        )

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.gap(x)        
        x = self.convblock8(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
