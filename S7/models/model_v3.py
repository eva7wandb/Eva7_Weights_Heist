# Model Definition -- I modified some of the contents of the earlier MNIST model
# This notebook defined with batch size of 64. We can increase it (to probably 256) to get a better result.

# padding = 1 is default, as we have to maintain the shape for all convolution blocks
# only last convolution block convblock8, made padding = 0 (Even there, since its an output layer, it doesn't matter, I guess)

# gap -- applied with kernel_size=4
# added argument, stride=stride inside conv_layer

# set regularization_off for all layers.

import torch.nn as nn
import torch.nn.functional as F

class depthwise_separable_block(nn.Module):
    def __init__(self, n_in, n_out, padding=1):
        super(depthwise_separable_block, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels=n_in, out_channels=n_in, bias=False,
            kernel_size=(3, 3), padding=1, groups=n_in,
        )
        self.pointwise = nn.Conv2d(
            in_channels=n_in, out_channels=n_out, bias=False,
            kernel_size=(1, 1),
        )
        self.depth_separable = nn.Sequential(
            self.depthwise,
            self.pointwise,
            nn.ReLU(),
            nn.BatchNorm2d(n_out),
        )
    def forward(self, x):
        out = self.depth_separable(x)
        return out
    
class dilated_conv_block(nn.Module):
    
    def __init__(self, n_in, n_mid, n_out, padding=1, dilation=2, stride=1, kernel_size=(3, 3)):
        super(dilated_conv_block, self).__init__()
        self.dilated = nn.Sequential(
            nn.Conv2d(
                in_channels=n_in, out_channels=n_mid,
                kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,),
            nn.Conv2d(
                in_channels=n_mid, out_channels=n_out,
                kernel_size=(1, 1), stride=1, padding=0, dilation=1),
            nn.ReLU(),
            nn.BatchNorm2d(n_out),
        )
    
    def forward(self, x):
        out = self.dilated(x)
        return out

class transition_block(nn.Module):
    
    def __init__(self, n_in, n_mid, n_out, padding=1, dilation=1, stride=2, kernel_size=(3, 3)):
        super(transition_block, self).__init__()
        self.transit = nn.Sequential(
            nn.Conv2d(in_channels=n_in, out_channels=n_mid,
                      kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,),
            nn.Conv2d(in_channels=n_mid, out_channels=n_out,kernel_size=(1, 1), stride=1, padding=0, dilation=dilation),
            nn.ReLU(),
            nn.BatchNorm2d(n_out),
        )
    
    def forward(self, x):
        out = self.transit(x)
        return out
        

class conv_block(nn.Module):
    def __init__(
        self,
        n_in, n_out, n_mid=32,
        kernel_size=(3, 3), stride=1, padding=1,
        bias=False, mid_channels=32,
        groups=1, dilation=1,
        activation=True,
        dropout_value=0.01,
        batch_norm=True,
        layer_norm=False,
        group_norm=False,
        group_norm_groups=2
    ):
        super().__init__()
        conv_layer1 = nn.Conv2d(
            in_channels=n_in, out_channels=n_mid, 
            kernel_size=kernel_size, padding=padding, 
            bias=bias, stride=stride, groups=groups, dilation=dilation,
        )
        conv_layer2 = nn.Conv2d(
            in_channels=n_mid, out_channels=n_out,
            kernel_size=kernel_size, padding=padding,
            bias=bias, stride=stride, groups=groups, dilation=dilation,
        )
        elements = [conv_layer1, conv_layer2]

        operations = []

        for idx, element in enumerate(elements):
        
            # Add first conv_layer
            operations.append(element)
            
            if activation:
                operations.append(nn.ReLU())
            elif idx==0:
                operations.append(nn.ReLU())
            
            ## regularizers
            if batch_norm:
                if idx==0:
                    operations.append(nn.BatchNorm2d(n_mid))
                elif idx==1:
                    operations.append(nn.BatchNorm2d(n_out))
            
            if layer_norm:
                if idx==0:
                    operations.append(nn.GroupNorm(1, n_mid))
                else:
                    operations.append(nn.GroupNorm(1, n_out))
          
            if group_norm:
                if idx==0:
                    operations.append(nn.GroupNorm(group_norm_groups, n_mid))
                else:
                    operations.append(nn.GroupNorm(group_norm_groups, n_out))
          
            if dropout_value:
                operations.append(nn.Dropout(dropout_value))
        
        self.layer = nn.Sequential(*operations)
    
    def forward(self, x):
        out = self.layer(x)
        return out

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
            'dropout_value': 0.0,
            'batch_norm': False,
            'layer_norm': False,
            'group_norm': False,
        }
        
        # CONVOLUTION BLOCK 1
        self.convblock1 = conv_block(n_in=3, n_mid=32, n_out=64, padding=1,)

        # TRANSITION BLOCK 1 | Dilated Kernel Convolution
        self.trans1 = transition_block(n_in=64, n_mid=32, n_out=32, stride=2, padding=1,)
        
        # CONVOLUTION BLOCK 2
        self.convblock2 = depthwise_separable_block(n_in=32, n_out=64, padding=1)
        

        # TRANSITION BLOCK 2 | Depthwise Separable Convolution
        self.trans2 = transition_block(n_in=64, n_mid=32, n_out=64, stride=2, padding=2,)

        # CONVOLUTION BLOCK 3
        self.convblock3 = conv_block(n_in=64, n_mid=32, n_out=32, padding=1,)
        self.convblock3_2 = dilated_conv_block(n_in=32, n_mid=32, n_out=64, padding=2,)

        # TRANSITION BLOCK 3
        self.trans3 = transition_block(n_in=64, n_mid=32, n_out=16, stride=2, padding=1,) 

        # CONVOLUTION BLOCK 4
        self.convblock4 = conv_block(n_in=16, n_mid=16, n_out=32, padding=1,)
        
        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0,),
            # nn.Dropout(dropout_value),
            nn.AvgPool2d(kernel_size=5),
        )

    def forward(self, x):
        x = self.convblock1(x)
        x = self.trans1(x)
        
        x = self.convblock2(x)
        x = self.trans2(x)

        x = self.convblock3(x)
        x = self.convblock3_2(x)
        x = self.trans3(x)

        x = self.convblock4(x)
        x = self.gap(x)
        
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
 