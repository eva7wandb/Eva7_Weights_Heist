## S12 - The Dawn of Transformers


## Team - Weights_Heist
### Team Members - 

| Name        | mail           |
|:-------------|:--------------|
|Gopinath Venkatesan|gops75[at]gmail[dot]com|
|Muhsin Mohammed|askmuhsin[at]gmail[dot]com|
|Rashu Tyagi|rashutyagi116[at]gmail[dot]com| 
|Subramanya R|subrananjun[at]gmail[dot]com| 


-----

# SPATIAL TRANSFORMER NETWORKS
[Spatial Transformer Networks paper](https://arxiv.org/pdf/1506.02025.pdf)    
[Pytorch tutorial for STN](https://brsoff.github.io/tutorials/intermediate/spatial_transformer_tutorial.html)    


Spatial transformer netowrk (STN) is a learnable module which is used with a CNN.     
It is used to address the limitation of CNNs in being not able to be spatially invariant to the input data.     
Essentially STN outputs an affine transformation of the input image, which will be enhanced with the appropriate scale, sheer, rotation and translations.     
The nice thing about STN is, it can be plugged into any existing CNN architectures.     
It also does not have to be limited to be used just as the first block of transfromation.

This image illustrates the implementation of STN 👇.     
As it can be noticed the final output of the STN is of the same size as the input image, it has undergone some affine transformations.    
The output of the Regressor is an affine matrix of size 2X3. This matrix parametrizes the affine transformations. (more on this in next section)     

![STN network](https://github.com/eva7wandb/Eva7_Weights_Heist/blob/main/S12/resources/Spatial_tranformer_network.drawio_bg.png)     



# AFFINE TRANSFORMATIONS
Affine transformations are mathematical operations that can be performed on an image to attain a desired output.    
Where the desired output can be some degree of Scale, Shear, Translation, and Rotation.    
An affine matrix can be a 2X3 matrix like -- 
```
1 0 0
0 1 0
```
in 👆 this case the output will be same as the input. each of the above values can be thought of as 👇. (note- rotation can be thought of as a combination of shear and scale on x and y axis)
```
Scale-X Shear-X Translate-X
Shear-Y Scale-y Translate-Y
```

here is how affine transformations be done on pytorch (given an input image and a affine matrix) -- 
```python
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch

def test_affine_transformations(theta, test_input):
    grid = F.affine_grid(theta, test_input.size())
    test_output = F.grid_sample(test_input, grid)
    
    plt.figure(figsize=(7, 7))
    plt.subplot(221)
    plt.imshow(grid[:, :, :, 0].detach().numpy().transpose((1, 2, 0)).squeeze())
    plt.subplot(222)
    plt.imshow(grid[:, :, :, 1].detach().numpy().transpose((1, 2, 0)).squeeze())
    plt.subplot(223)
    plt.imshow(test_input.squeeze().detach().numpy())
    plt.subplot(224)
    plt.imshow(test_output.squeeze().detach().numpy())
    plt.tight_layout()

theta = torch.tensor([
    [
        [1., 0., 0.2],
        [0., 1., 0.]
    ]
], dtype=torch.float)
test_affine_transformations(theta, test_input)
```
![affine implementation pytorch](https://github.com/eva7wandb/Eva7_Weights_Heist/blob/main/S12/resources/affine_transform_sample.png)     

# CIFAR10 trained on STN
Training Notebook -- <strong>[STN CIFAR 10](https://github.com/eva7wandb/Eva7_Weights_Heist/blob/main/S12/STN_CIFAR10.ipynb)</strong>    
## Visualize STN
The outputs of STN visualized as an animation starting from EPOCH 1 to EPOCH 50. 👇
<img src="https://github.com/eva7wandb/Eva7_Weights_Heist/blob/main/S12/stn_cifar10.gif" data-canonical-src="https://github.com/eva7wandb/Eva7_Weights_Heist/blob/main/S12/stn_cifar10.gif" width="600" height="400" />


## Model Summary
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 26, 26]           9,472
         MaxPool2d-2           [-1, 64, 13, 13]               0
              ReLU-3           [-1, 64, 13, 13]               0
            Conv2d-4            [-1, 128, 9, 9]         204,928
         MaxPool2d-5            [-1, 128, 4, 4]               0
              ReLU-6            [-1, 128, 4, 4]               0
            Linear-7                  [-1, 256]         524,544
              ReLU-8                  [-1, 256]               0
            Linear-9                    [-1, 6]           1,542
           Conv2d-10           [-1, 10, 28, 28]             760
           Conv2d-11           [-1, 20, 10, 10]           5,020
        Dropout2d-12           [-1, 20, 10, 10]               0
           Linear-13                   [-1, 50]          25,050
           Linear-14                   [-1, 10]             510
================================================================
Total params: 771,826
Trainable params: 771,826
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 0.70
Params size (MB): 2.94
Estimated Total Size (MB): 3.66
----------------------------------------------------------------
```
## Loss Graph
![image](https://user-images.githubusercontent.com/8600096/147463131-630c9744-d2b0-4f43-b1c5-ab708cbb2f63.png)


## Training logs
```
[INFO] Begin training for 50 epochs.
TRAIN Epoch:0 Loss:2.1503 Batch:1562 Acc:19.05: 100%|██████████| 1563/1563 [00:38<00:00, 40.84it/s]
TEST         Loss:1.8884         Acc:32.43         [3243 / 10000]
TRAIN Epoch:1 Loss:2.0797 Batch:1562 Acc:25.95: 100%|██████████| 1563/1563 [00:38<00:00, 40.44it/s]
TEST         Loss:1.7600         Acc:36.74         [3674 / 10000]
TRAIN Epoch:2 Loss:1.8817 Batch:1562 Acc:29.53: 100%|██████████| 1563/1563 [00:39<00:00, 39.41it/s]
TEST         Loss:1.6496         Acc:39.85         [3985 / 10000]
TRAIN Epoch:4 Loss:1.6398 Batch:1562 Acc:33.03: 100%|██████████| 1563/1563 [00:37<00:00, 41.57it/s]
TEST         Loss:1.5573         Acc:43.70         [4370 / 10000]
TRAIN Epoch:5 Loss:1.6732 Batch:1562 Acc:34.81: 100%|██████████| 1563/1563 [00:41<00:00, 37.46it/s]
TEST         Loss:1.5358         Acc:44.45         [4445 / 10000]
TRAIN Epoch:6 Loss:1.8877 Batch:1562 Acc:35.55: 100%|██████████| 1563/1563 [00:39<00:00, 39.09it/s]
TEST         Loss:1.5573         Acc:43.20         [4320 / 10000]
TRAIN Epoch:7 Loss:1.9346 Batch:1562 Acc:36.51: 100%|██████████| 1563/1563 [00:40<00:00, 38.64it/s]
TEST         Loss:1.4723         Acc:46.69         [4669 / 10000]
TRAIN Epoch:8 Loss:1.684 Batch:1562 Acc:37.17: 100%|██████████| 1563/1563 [00:40<00:00, 38.86it/s] 
TEST         Loss:1.4686         Acc:46.40         [4640 / 10000]
TRAIN Epoch:9 Loss:1.6618 Batch:1562 Acc:37.37: 100%|██████████| 1563/1563 [00:36<00:00, 42.26it/s]
TEST         Loss:1.4327         Acc:47.63         [4763 / 10000]
TRAIN Epoch:10 Loss:1.5915 Batch:1562 Acc:37.72: 100%|██████████| 1563/1563 [00:39<00:00, 39.47it/s]
TEST         Loss:1.4305         Acc:48.11         [4811 / 10000]
TRAIN Epoch:11 Loss:1.4979 Batch:1562 Acc:38.96: 100%|██████████| 1563/1563 [00:36<00:00, 42.46it/s]
TEST         Loss:1.4049         Acc:49.01         [4901 / 10000]
TRAIN Epoch:12 Loss:1.6643 Batch:1562 Acc:39.01: 100%|██████████| 1563/1563 [00:40<00:00, 38.62it/s]
TEST         Loss:1.3773         Acc:50.49         [5049 / 10000]
TRAIN Epoch:13 Loss:1.8085 Batch:1562 Acc:39.93: 100%|██████████| 1563/1563 [00:39<00:00, 39.77it/s]
TEST         Loss:1.3792         Acc:50.30         [5030 / 10000]
TRAIN Epoch:14 Loss:1.8097 Batch:1562 Acc:40.37: 100%|██████████| 1563/1563 [00:40<00:00, 38.27it/s]
TEST         Loss:1.3827         Acc:50.24         [5024 / 10000]
TRAIN Epoch:15 Loss:1.861 Batch:1562 Acc:40.44: 100%|██████████| 1563/1563 [00:38<00:00, 40.49it/s] 
TEST         Loss:1.3599         Acc:50.85         [5085 / 10000]
TRAIN Epoch:16 Loss:1.826 Batch:1562 Acc:40.77: 100%|██████████| 1563/1563 [00:41<00:00, 37.46it/s] 
TEST         Loss:1.3284         Acc:51.82         [5182 / 10000]
TRAIN Epoch:17 Loss:1.5711 Batch:1562 Acc:41.31: 100%|██████████| 1563/1563 [00:41<00:00, 38.12it/s]
TEST         Loss:1.3357         Acc:52.40         [5240 / 10000]
TRAIN Epoch:18 Loss:1.8055 Batch:1562 Acc:41.31: 100%|██████████| 1563/1563 [00:39<00:00, 39.97it/s]
TEST         Loss:1.3109         Acc:52.04         [5204 / 10000]
TRAIN Epoch:19 Loss:2.1209 Batch:1562 Acc:41.69: 100%|██████████| 1563/1563 [00:39<00:00, 39.97it/s]
TEST         Loss:1.3357         Acc:51.38         [5138 / 10000]
TRAIN Epoch:20 Loss:1.4595 Batch:1562 Acc:41.94: 100%|██████████| 1563/1563 [00:40<00:00, 39.01it/s]
TEST         Loss:1.2897         Acc:53.47         [5347 / 10000]
TRAIN Epoch:21 Loss:1.668 Batch:1562 Acc:42.08: 100%|██████████| 1563/1563 [00:38<00:00, 40.43it/s] 
TEST         Loss:1.2864         Acc:53.86         [5386 / 10000]
TRAIN Epoch:22 Loss:1.3464 Batch:1562 Acc:42.69: 100%|██████████| 1563/1563 [00:38<00:00, 40.29it/s]
TEST         Loss:1.3325         Acc:51.25         [5125 / 10000]
TRAIN Epoch:23 Loss:1.9882 Batch:1562 Acc:42.24: 100%|██████████| 1563/1563 [00:38<00:00, 41.11it/s]
TEST         Loss:1.3185         Acc:53.00         [5300 / 10000]
TRAIN Epoch:24 Loss:1.4191 Batch:1562 Acc:42.85: 100%|██████████| 1563/1563 [00:40<00:00, 38.96it/s]
TEST         Loss:1.2950         Acc:53.52         [5352 / 10000]
TRAIN Epoch:25 Loss:1.8104 Batch:1562 Acc:42.94: 100%|██████████| 1563/1563 [00:43<00:00, 36.04it/s]
TEST         Loss:1.2936         Acc:53.03         [5303 / 10000]
TRAIN Epoch:26 Loss:1.1478 Batch:1562 Acc:42.55: 100%|██████████| 1563/1563 [00:42<00:00, 37.15it/s]
TEST         Loss:1.2518         Acc:54.10         [5410 / 10000]
TRAIN Epoch:27 Loss:1.429 Batch:1562 Acc:43.26: 100%|██████████| 1563/1563 [00:42<00:00, 37.15it/s] 
TEST         Loss:1.2710         Acc:53.71         [5371 / 10000]
TRAIN Epoch:28 Loss:1.5259 Batch:1562 Acc:43.68: 100%|██████████| 1563/1563 [00:42<00:00, 36.83it/s]
TEST         Loss:1.2764         Acc:54.96         [5496 / 10000]
TRAIN Epoch:29 Loss:2.4441 Batch:1562 Acc:42.88: 100%|██████████| 1563/1563 [00:41<00:00, 37.96it/s]
TEST         Loss:1.3091         Acc:53.15         [5315 / 10000]
TRAIN Epoch:30 Loss:1.6018 Batch:1562 Acc:42.89: 100%|██████████| 1563/1563 [00:39<00:00, 39.12it/s]
TEST         Loss:1.2891         Acc:54.29         [5429 / 10000]
TRAIN Epoch:31 Loss:1.5309 Batch:1562 Acc:43.08: 100%|██████████| 1563/1563 [00:41<00:00, 37.90it/s]
TEST         Loss:1.2735         Acc:54.24         [5424 / 10000]
TRAIN Epoch:32 Loss:1.527 Batch:1562 Acc:42.50: 100%|██████████| 1563/1563 [00:41<00:00, 37.64it/s] 
TEST         Loss:1.2616         Acc:54.24         [5424 / 10000]
TRAIN Epoch:33 Loss:1.5706 Batch:1562 Acc:43.04: 100%|██████████| 1563/1563 [00:42<00:00, 37.09it/s]
TEST         Loss:1.2691         Acc:54.22         [5422 / 10000]
TRAIN Epoch:34 Loss:1.3107 Batch:1562 Acc:43.49: 100%|██████████| 1563/1563 [00:41<00:00, 37.27it/s]
TEST         Loss:1.2627         Acc:54.15         [5415 / 10000]
TRAIN Epoch:35 Loss:1.3429 Batch:1562 Acc:43.25: 100%|██████████| 1563/1563 [00:40<00:00, 38.15it/s]
TEST         Loss:1.2448         Acc:55.48         [5548 / 10000]
TRAIN Epoch:36 Loss:1.4635 Batch:1562 Acc:43.75: 100%|██████████| 1563/1563 [00:40<00:00, 38.85it/s]
TEST         Loss:1.2864         Acc:54.07         [5407 / 10000]
TRAIN Epoch:37 Loss:1.4084 Batch:1562 Acc:43.55: 100%|██████████| 1563/1563 [00:42<00:00, 37.20it/s]
TEST         Loss:1.3375         Acc:51.93         [5193 / 10000]
TRAIN Epoch:38 Loss:1.7473 Batch:1562 Acc:43.19: 100%|██████████| 1563/1563 [00:37<00:00, 41.19it/s]
TEST         Loss:1.2978         Acc:53.63         [5363 / 10000]
TRAIN Epoch:39 Loss:1.276 Batch:1562 Acc:43.55: 100%|██████████| 1563/1563 [00:38<00:00, 41.09it/s] 
TEST         Loss:1.2535         Acc:54.67         [5467 / 10000]
TRAIN Epoch:40 Loss:1.5751 Batch:1562 Acc:43.90: 100%|██████████| 1563/1563 [00:37<00:00, 41.43it/s]
TEST         Loss:1.2550         Acc:54.50         [5450 / 10000]
TRAIN Epoch:41 Loss:1.4187 Batch:1562 Acc:43.66: 100%|██████████| 1563/1563 [00:37<00:00, 41.71it/s]
TEST         Loss:1.2477         Acc:55.06         [5506 / 10000]
TRAIN Epoch:42 Loss:1.5312 Batch:1562 Acc:43.77: 100%|██████████| 1563/1563 [00:37<00:00, 42.15it/s]
TEST         Loss:1.2735         Acc:54.38         [5438 / 10000]
TRAIN Epoch:43 Loss:1.4198 Batch:1562 Acc:42.70: 100%|██████████| 1563/1563 [00:37<00:00, 41.64it/s]
TEST         Loss:1.3194         Acc:53.17         [5317 / 10000]
TRAIN Epoch:44 Loss:1.2714 Batch:1562 Acc:43.99: 100%|██████████| 1563/1563 [00:37<00:00, 41.20it/s]
TEST         Loss:1.2597         Acc:54.14         [5414 / 10000]
TRAIN Epoch:45 Loss:1.3426 Batch:1562 Acc:43.56: 100%|██████████| 1563/1563 [00:37<00:00, 42.01it/s]
TEST         Loss:1.3054         Acc:52.89         [5289 / 10000]
TRAIN Epoch:46 Loss:1.7026 Batch:1562 Acc:43.73: 100%|██████████| 1563/1563 [00:37<00:00, 41.82it/s]
TEST         Loss:1.2461         Acc:55.80         [5580 / 10000]
TRAIN Epoch:47 Loss:1.5558 Batch:1562 Acc:43.79: 100%|██████████| 1563/1563 [00:38<00:00, 40.49it/s]
TEST         Loss:1.2587         Acc:55.29         [5529 / 10000]
TRAIN Epoch:48 Loss:1.7813 Batch:1562 Acc:43.69: 100%|██████████| 1563/1563 [00:38<00:00, 40.72it/s]
TEST         Loss:1.3151         Acc:53.06         [5306 / 10000]
TRAIN Epoch:49 Loss:1.2152 Batch:1562 Acc:44.53: 100%|██████████| 1563/1563 [00:38<00:00, 40.73it/s]
TEST         Loss:1.2285         Acc:56.46         [5646 / 10000]
```
