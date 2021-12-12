Q1 -- model.py
```
'''ResNet in PyTorch.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, 
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        # self.shortcut = nn.Sequential()
        # if stride != 1 or in_planes != self.expansion*planes:
        #     self.shortcut = nn.Sequential(
        #         nn.Conv2d(
        #             in_planes, self.expansion*planes,
        #             kernel_size=1, stride=stride, bias=False
        #         ),
        #         nn.BatchNorm2d(self.expansion*planes)
        #     )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # out += self.shortcut(x)
        out = F.relu(out)
        return out

class XBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(XBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, 
            stride=stride, padding=1, bias=False
        )
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.bn1 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = self.bn1(self.maxpool1(self.conv1(x)))
        out = F.relu(out)
        return out

class CustResNet(nn.Module):
    def __init__(self, block, xblock, num_blocks, num_classes=10):
        super(CustResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1_x = self._make_layer(xblock, 128, num_blocks[0], stride=1)
        self.layer1_R1 = self._make_layer(block, 128, num_blocks[0], stride=1)

        self.layer2 = self._make_layer(xblock, 256, num_blocks[1], stride=1)

        self.layer3_x = self._make_layer(xblock, 512, num_blocks[2], stride=1)
        self.layer3_R2 = self._make_layer(block, 512, num_blocks[2], stride=1)

        #self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        #prep layer
        out = F.relu(self.bn1(self.conv1(x)))
        #layer 1
        out = self.layer1_x(out)
        out = out + self.layer1_R1(out)
        #layer 2
        out = self.layer2(out)
        #layer 3
        out = self.layer3_x(out)
        out = out + self.layer3_R2(out) 
        #out = self.layer4(out)
        out = F.max_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return F.log_softmax(out, dim=-1)


def CustomResNet():
    return CustResNet(BasicBlock, XBlock, [1, 1, 1])

def test():
    net = CustomResNet()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())
```

Q2 -- main 
```
from utils import (
    setup, data, viz
)
from torch_lr_finder import LRFinder
from utils.training import train
from utils.testing import test

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

setup.set_seed()
cuda = setup.is_cuda()
device = setup.get_device()


class Trainer:
    def __init__(
        self, model,
        lr=0.01,
        batch_size=128,
        scheduler = 'ReduceLROnPlateau',  #values are CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau
        model_viz=True,
        model_path=None,
        eval_model_on_load=True,
        label_smoothing=0.0,
        optimizer='SGD',
        run_find_lr=False
    ):
        print(f"[INFO] Loading Data")
        self.train_loader = data.CIFAR10_dataset(
            train=True, cuda=cuda
        ).get_loader(batch_size)
        self.test_loader = data.CIFAR10_dataset(
            train=False, cuda=cuda
        ).get_loader(batch_size)
        self.test_loader_unnormalized = data.CIFAR10_dataset(
            train=False, cuda=cuda, normalize=False
        ).get_loader(batch_size)

        self.net = model.to(device)
        if model_viz:
            viz.show_model_summary(self.net)
        
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        if optimizer=='SGD':
            self.optimizer = optim.SGD(
                self.net.parameters(), lr=self.lr,
                momentum=0.9, weight_decay=1e-4
            )
        elif optimizer=='Adam':
            self.optimizer = optim.Adam(
                self.net.parameters(), lr=self.lr, 
                betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False
            )
        else:
            raise ValueError(f'{optimizer} is not valid choice. Please select one of valid scheduler - SGD, Adam')
        
        print(self.optimizer)
        print('-' * 64)
        if run_find_lr:
            print('Running LR finder ... ')
            self.find_lr()
        print('-' * 64)
        
        if scheduler == 'CosineAnnealingLR':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)
        elif scheduler == 'ReduceLROnPlateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        elif scheduler == 'OneCycleLR':
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=lr, steps_per_epoch=len(self.train_loader), 
                epochs=24, div_factor=10, 
                final_div_factor=1, pct_start=0.2, 
                three_phase=False, anneal_strategy='linear'
            )
        else:
            raise ValueError(f'{scheduler} is not valid choice. Please select one of valid scheduler - CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau')

        self.logs = []
        self.lr_logs = []
        
        self.model_path = model_path
        if self.model_path:
            self.load_model()
        
        if eval_model_on_load:
            self.evaluate_model()
    
    def load_model(self):
        print("[INFO] Loading model to from path {}".format(self.model_path))
        self.net.load_state_dict(torch.load(self.model_path))
        
    def train_model(self, epochs, wandb=None):
        EPOCHS = epochs
        print(f"[INFO] Begin training for {EPOCHS} epochs.")
        
        for epoch in range(EPOCHS):
            lr_at_start_of_epoch = self.optimizer.param_groups[0]['lr']
            
            train_batch_loss, train_batch_acc, train_batch_lrs = train(
                self.net, device, 
                self.train_loader, self.optimizer, self.criterion, epoch, self.scheduler,
            )
            train_loss = np.mean(train_batch_loss)
            train_acc = np.mean(train_batch_acc)
            test_loss, test_acc = test(
                self.net, device,
                self.test_loader, self.criterion, epoch,
            )
            self.lr_logs.extend(train_batch_lrs)
            #self.scheduler.step(test_loss)
            
            ## logging
            log_temp = {
                "train_acc": train_acc,
                "test_acc": test_acc,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "lr": lr_at_start_of_epoch,
            }
            try:
                wandb.log(log_temp)
            except:
                pass
            
            self.logs.append(log_temp)
    
    def evaluate_model(self):
        test_loss, test_acc = test(
            self.net, device,
            self.test_loader, self.criterion, epoch=0,
        )
    
    def find_lr(self):
        lr_finder = LRFinder(self.net, self.optimizer, self.criterion, device=device)
        lr_finder.range_test(self.train_loader, end_lr=1, num_iter=100, step_mode="exp")
        lr_finder.plot()
        best_loss_ind = lr_finder.history['loss'].index(lr_finder.best_loss)
        print('few steps before and after best_loss')
        print('(relative step, lr, loss)')
        print(*[
            (n - 5, round(lr, 5), round(loss, 5))
            for n, (lr, loss) in enumerate(zip(
                lr_finder.history['lr'][best_loss_ind - 5 : best_loss_ind + 5], 
                lr_finder.history['loss'][best_loss_ind - 5 : best_loss_ind + 5], 
            ))
        ], sep='\n')
        self.lr_history = lr_finder.history
        lr_finder.reset()


def show_misclassification(trainer, cam_layer_name='layer4'):
    from utils.viz import visualize_sample
    from utils.testing import get_sample_predictions
    
    sample_preds = get_sample_predictions(trainer)
    
    for class_, samples in sample_preds['mistakes'].items():
        for sample in samples[:2]:
            visualize_sample(trainer, sample, cam_layer_name)

def show_loss_curves(logs):
    from utils.viz import visualize_loss

    visualize_loss(logs)
 ```
 
 Q3 -- log
 ```
 
[INFO] Begin training for 24 epochs.

TRAIN Epoch:0 Loss:1.1918 Batch:97 Acc:49.28: 100%|██████████| 98/98 [00:20<00:00,  4.74it/s]

TEST         Loss:1.3077         Acc:60.11         [6011 / 10000]

TRAIN Epoch:1 Loss:1.027 Batch:97 Acc:67.80: 100%|██████████| 98/98 [00:20<00:00,  4.67it/s] 

TEST         Loss:1.1525         Acc:67.35         [6735 / 10000]

TRAIN Epoch:2 Loss:0.8958 Batch:97 Acc:75.25: 100%|██████████| 98/98 [00:21<00:00,  4.52it/s]

TEST         Loss:0.9657         Acc:75.00         [7500 / 10000]

TRAIN Epoch:3 Loss:0.8899 Batch:97 Acc:78.86: 100%|██████████| 98/98 [00:22<00:00,  4.44it/s]

TEST         Loss:0.8641         Acc:79.01         [7901 / 10000]

TRAIN Epoch:4 Loss:0.8544 Batch:97 Acc:80.52: 100%|██████████| 98/98 [00:21<00:00,  4.52it/s]

TEST         Loss:0.8460         Acc:79.48         [7948 / 10000]

TRAIN Epoch:5 Loss:0.7655 Batch:97 Acc:83.67: 100%|██████████| 98/98 [00:21<00:00,  4.57it/s]

TEST         Loss:0.7945         Acc:81.59         [8159 / 10000]

TRAIN Epoch:6 Loss:0.71 Batch:97 Acc:86.06: 100%|██████████| 98/98 [00:21<00:00,  4.61it/s]  

TEST         Loss:0.7110         Acc:84.50         [8450 / 10000]

TRAIN Epoch:7 Loss:0.6771 Batch:97 Acc:87.69: 100%|██████████| 98/98 [00:21<00:00,  4.46it/s]

TEST         Loss:0.7014         Acc:85.43         [8543 / 10000]

TRAIN Epoch:8 Loss:0.5373 Batch:97 Acc:88.88: 100%|██████████| 98/98 [00:22<00:00,  4.41it/s]

TEST         Loss:0.6130         Acc:88.41         [8841 / 10000]

TRAIN Epoch:9 Loss:0.5551 Batch:97 Acc:90.57: 100%|██████████| 98/98 [00:23<00:00,  4.23it/s]

TEST         Loss:0.6067         Acc:88.58         [8858 / 10000]

TRAIN Epoch:10 Loss:0.5537 Batch:97 Acc:91.52: 100%|██████████| 98/98 [00:21<00:00,  4.46it/s]

TEST         Loss:0.6033         Acc:89.02         [8902 / 10000]

TRAIN Epoch:11 Loss:0.5539 Batch:97 Acc:92.14: 100%|██████████| 98/98 [00:22<00:00,  4.38it/s]

TEST         Loss:0.5590         Acc:90.21         [9021 / 10000]

TRAIN Epoch:12 Loss:0.5273 Batch:97 Acc:93.06: 100%|██████████| 98/98 [00:22<00:00,  4.40it/s]

TEST         Loss:0.5799         Acc:89.11         [8911 / 10000]

TRAIN Epoch:13 Loss:0.5053 Batch:97 Acc:94.03: 100%|██████████| 98/98 [00:22<00:00,  4.35it/s]

TEST         Loss:0.6791         Acc:85.98         [8598 / 10000]

TRAIN Epoch:14 Loss:0.4878 Batch:97 Acc:94.57: 100%|██████████| 98/98 [00:22<00:00,  4.39it/s]

TEST         Loss:0.5876         Acc:89.50         [8950 / 10000]

TRAIN Epoch:15 Loss:0.4486 Batch:97 Acc:94.98: 100%|██████████| 98/98 [00:23<00:00,  4.21it/s]

TEST         Loss:0.5354         Acc:91.06         [9106 / 10000]

TRAIN Epoch:16 Loss:0.3981 Batch:97 Acc:95.70: 100%|██████████| 98/98 [00:23<00:00,  4.18it/s]

TEST         Loss:0.5262         Acc:91.54         [9154 / 10000]

TRAIN Epoch:17 Loss:0.4019 Batch:97 Acc:96.24: 100%|██████████| 98/98 [00:23<00:00,  4.22it/s]

TEST         Loss:0.5128         Acc:92.12         [9212 / 10000]

TRAIN Epoch:18 Loss:0.4336 Batch:97 Acc:96.85: 100%|██████████| 98/98 [00:23<00:00,  4.19it/s]

TEST         Loss:0.5073         Acc:92.16         [9216 / 10000]

TRAIN Epoch:19 Loss:0.3852 Batch:97 Acc:97.22: 100%|██████████| 98/98 [00:24<00:00,  3.98it/s]

TEST         Loss:0.5067         Acc:92.19         [9219 / 10000]

TRAIN Epoch:20 Loss:0.3764 Batch:97 Acc:97.62: 100%|██████████| 98/98 [00:23<00:00,  4.08it/s]

TEST         Loss:0.4821         Acc:92.82         [9282 / 10000]

TRAIN Epoch:21 Loss:0.3584 Batch:97 Acc:97.96: 100%|██████████| 98/98 [00:24<00:00,  4.03it/s]

TEST         Loss:0.4847         Acc:93.02         [9302 / 10000]

TRAIN Epoch:22 Loss:0.3578 Batch:97 Acc:98.35: 100%|██████████| 98/98 [00:23<00:00,  4.12it/s]

TEST         Loss:0.4746         Acc:93.16         [9316 / 10000]

TRAIN Epoch:23 Loss:0.3549 Batch:97 Acc:98.63: 100%|██████████| 98/98 [00:23<00:00,  4.09it/s]

TEST         Loss:0.4679         Acc:93.53         [9353 / 10000]
```

Q4 -- image aug
```
        self.mean = (0.4890062, 0.47970363, 0.47680542)
        self.std = (0.264582, 0.258996, 0.25643882)
        
        if normalize:
            self.train_transforms = A.Compose([
                A.Sequential([
                    A.PadIfNeeded(40,40),
                    A.RandomCrop(32,32)],
                    p=0.75
                ),
                #A.CropAndPad(px=4,keep_size=False, p=0.5,),
                #A.RandomCrop(32, 32, always_apply=False, p=1),
                A.HorizontalFlip(p=0.75),
                #A.Cutout (num_holes=8, max_h_size=8, fill_value=(0.491, 0.482, 0.447), always_apply=False, p=0.5),
                A.Cutout(num_holes=1, max_h_size=8, max_w_size=8, fill_value=.4, always_apply=False, p=0.75),
#                 A.CoarseDropout(
#                     max_holes=3, max_height=16, max_width=16, min_holes=None, min_height=None, min_width=None, 
#                     fill_value=self.mean, mask_fill_value=None, always_apply=False, p=0.25
#                 ),
#                 A.Rotate(limit=5, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5),
                A.Normalize(
                    mean=self.mean, 
                    std=self.std,
                    always_apply=True
                ),
                ToTensorV2()
            ])
            self.test_transforms = A.Compose([
                A.Normalize(
                    mean=self.mean, 
                    std=self.std,
                    always_apply=True
                ),
                ToTensorV2()
            ])
        else:
            self.train_transforms = A.Compose([
                A.RandomCrop(32, 32, always_apply=False, p=0.5),
                A.CoarseDropout(
                    max_holes=3, max_height=16, max_width=16, min_holes=None, min_height=None, min_width=None, 
                    fill_value=self.mean, mask_fill_value=None, always_apply=False, p=0.25
                ),
                A.Rotate(limit=5, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5),
                ToTensorV2()
            ])
            self.test_transforms = A.Compose([
                ToTensorV2()
            ])
```

Q5 -- readme

```
Project README -- 

https://github.com/eva7wandb/Eva7_Weights_Heist/blob/main/S9/README.md




main repo README --

https://github.com/eva7wandb/Weights_Heist_Flow/blob/main/README.md
```


Q6 -- package
```
https://github.com/eva7wandb/Weights_Heist_Flow
```
 
 
    
 
