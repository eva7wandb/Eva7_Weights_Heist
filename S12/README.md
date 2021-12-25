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
It is used to address the limitation of CNNs in being able to be spatially invariant to the input data.     
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

## Training logs
## visalize STN
## Comparison with and without STN






