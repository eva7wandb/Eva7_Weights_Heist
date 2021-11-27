import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomConvLayer(nn.Module):
    def __init__(
        self,
        in_channels, out_channels,
        kernel_size=3, padding=1, stride=1, bias=False,
        max_pool_kernel=2,
        
        use_mp=False,
        use_bn=True,
        use_relu=True,
    ):
        super().__init__()
        
        conv_layer = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, 
            kernel_size=kernel_size, padding=padding, stride=stride,
            bias=bias
        )
        
        elements = [conv_layer]
        
        if use_mp:
            max_pool = nn.MaxPool2d(max_pool_kernel)
            elements.append(max_pool)
            
        if use_bn:
            batch_norm = nn.BatchNorm2d(out_channels)
            elements.append(batch_norm)
            
        if use_relu:
            relu = nn.ReLU()
            elements.append(relu)
            
        self.layer = nn.Sequential(*elements)
        
    def forward(self, x):
        x = self.layer(x)
        return x
    
    
class ResNet9(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.block_0 = CustomConvLayer(3, 64)
        
        self.block_1_m = CustomConvLayer(64, 128, use_mp=True)
        self.block_1_r = nn.Sequential(
            CustomConvLayer(128, 128),
            CustomConvLayer(128, 128)
        )
        
        self.block_2 = CustomConvLayer(128, 256, use_mp=True)
        
        self.block_3_m = CustomConvLayer(256, 512, use_mp=True)
        self.block_3_r = nn.Sequential(
            CustomConvLayer(512, 512),
            CustomConvLayer(512, 512)
        )
        
        self.block_4 = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Conv2d(512, 10, kernel_size=1)
        )
    
    def forward(self, x):
        ## prep block
        x = self.block_0(x)
        ## block 1
        x = self.block_1_m(x)
        r1 = self.block_1_r(x)
        x = x + r1
        ## block 2
        x = self.block_2(x)
        ## block 3
        x = self.block_3_m(x)
        r2 = self.block_3_r(x)
        x = x + r2
        ## block 4
        x = self.block_4(x)
        x = x.view(-1, 10)
        
        return F.log_softmax(x, dim=-1)
 