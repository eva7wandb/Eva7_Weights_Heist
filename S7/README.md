# S7 - Advanced Concepts

## Team - Weights_Heist
## Team Members - 

| Name        | mail           |
| ------------- |:-------------:|
|Gopinath Venkatesan|gops75[at]gmail[dot]com|
|Muhsin Mohammed|askmuhsin[at]gmail[dot]com|
|Rashu Tyagi|rashutyagi116[at]gmail[dot]com| 
|Subramanya R|subrananjun[at]gmail[dot]com| 

In this tasks we achieved 85% + accuracy on CIFAR10 using 181,322 parameters.

## Model Summary
![image](https://user-images.githubusercontent.com/8600096/141836102-6183a32c-97cc-4154-9771-bd0f2c8edaae.png)

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

