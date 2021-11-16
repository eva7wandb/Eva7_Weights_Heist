# S7 - Advanced Concepts

## Team - Weights_Heist
## Team Members - 

| Name        | mail           |
| ------------- |:-------------:|
|Gopinath Venkatesan|gops75[at]gmail[dot]com|
|Muhsin Mohammed|askmuhsin[at]gmail[dot]com|
|Rashu Tyagi|rashutyagi116[at]gmail[dot]com| 
|Subramanya R|subrananjun[at]gmail[dot]com| 

* Objective: To devise a custom convolution neural network (CNN) architecture to classify multi-class image data, and apply it to CIFAR-10 dataset for performance evaluation.    
* Result: In this tasks we achieved 85% + accuracy on CIFAR10 using 181,322 parameters.

## Model Summary
The skeleton structure for the architecture is fixed which is a sequence containing 4 Convolution blocks and an Output block, denoted as C1-C2-C3-C4-O, and it is required to have transition blocks between any two Convolution blocks (with stride = 2) -- essentially to replace the MaxPooling layers.

Note: Unless explicitly mentioned, all the below blocks have ReLU() and BatchNorm() layers included before passing their output to another block.

> C1 -- Regular Convolution Block stacked with a pair of K = (3, 3) Convolution layers, s = 1, p = 1. Input shape is maintained. Input and Output Channels: (n_in, n_mid) and (n_mid and n_out). Input and Output shape: (C = 3, 32, 32), (N1, 32, 32)

> T1 -- Transition Block 1 is a stack of pair of Convolution layers, first layer with K = (3, 3), s = 2, p = 1, followed by a Pointwise Convolution layer. Shape is halved. Input and Output channels: (n_in, n_mid), and (n_mid, n_out). Input and Output shape: (N1, 32, 32), (N2, 16, 16)

> C2 -- Depthwise Separable (Convolution Block 2) Block performs a Depthwise Convolution (K = (3, 3), s = 1, p = 1, groups = n_in) followed by a Pointwise Convolution, K = (1, 1). Input and Output Channels: (n_in, n_in) and (n_in, n_out). Input and Output shape: (N2, 16, 16), (N3, 16, 16)

> T2 -- Transition Block 2: similar to T1 but with p = 2 to save shape loss, Input and Output shape: (N3, 16, 16), (N4, 8, 8)

> C3 -- Dilated Kernel Convolution Block 3, consists of two blocks, first block is a regular convolution block having two K = (3, 3) convolution layers, and the second block is stacked with a pair of convolution layers of which first layer performs the dilation and the second layer is a Pointwise Convolution. First block uses padding p = 1 retaining the shape. Input and Output shape: (N4, 8, 8), (N5, 8, 8). Second block uses Dilation=2 but uses p = 2 to gain shape (introduced to not to lose the shape). Input and Output shape: (N5, 8, 8), (N6, 9, 9).

> T3 -- Transition Block 3: similar to T1 or T2, with p = 1. Reduces shape. Input and Output shape: (N6, 9, 9), (N7, 5, 5)

> C4 -- Regular Convolution Block similar to C1, with p = 1. Retains shape. Input and Output shape: (N7, 5, 5), (N8, 5, 5)

> O -- Output Block consists of a Pointwise Convolution to bring the channel size to number of categories followed by Global Average Pooling (GAP) layer with a kernel size of 5. Pointwise Layer: Input and Output shape: (N8, 5, 5), (10, 5, 5); GAP Input and Output shape: (10, 5, 5), (10, 1, 1).

![image](https://user-images.githubusercontent.com/8600096/141836102-6183a32c-97cc-4154-9771-bd0f2c8edaae.png)


## Image Augmentations
For image augmenation we used the albumentations library since it is much faster than the pytorch native transforms.
```python
train_transform = A.Compose(
    [
        A.HorizontalFlip(),
        A.ShiftScaleRotate(),
        A.CoarseDropout(
            max_holes = 1, max_height=16, max_width=16, min_holes = 1, min_height=16,
            min_width=1, fill_value=mean, mask_fill_value = None
        ),
        A.Normalize(mean, std),
        A.pytorch.ToTensorV2(),
     ]
)
```

## Traing Log 
 - Final Model Summary -- 
![image](https://user-images.githubusercontent.com/8600096/141836450-55b77603-9dd9-4c8e-be35-7cc7bc71b943.png)

## Discussion
We had trained several models starting for 82k parameters to 181k parameters.    
Also with different Learning Rates, Reglarizations, and different types of convolution layers.     

Our architecture has 4 conv Blocks and an output (C1C2C3C40).     
We have not used any pooling layers. instead we have used a combination of strided covolutions, and dilated conv blocks.
In our second block we have used a Depthwise seperable convolution operation.   
And in the final output layer we used a GAP layer and no Dense Layers after it.

![image](https://user-images.githubusercontent.com/8600096/141837680-5f0000d0-3eec-4945-b5b2-8e642e0e35a1.png)
![image](https://user-images.githubusercontent.com/8600096/141837834-bdf0d8dc-d654-48e2-aae8-a1dbe996046b.png)
![image](https://user-images.githubusercontent.com/8600096/141837976-be2f9c42-0e8e-4456-9284-d0fb236b7d75.png)
![image](https://user-images.githubusercontent.com/8600096/141837895-e63618ac-6d00-4a52-8ba0-3428c8392bde.png)
![image](https://user-images.githubusercontent.com/8600096/141838009-f7d8e1ec-751c-4f25-b9d0-fe9c95b7c381.png)

