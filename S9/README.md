## S9 - Resnet and Higher Receptive Fields
&nbsp;
### `Objective: Devise a custom ResNet architecture for CIFAR-10 (the details of which is shown in the figure below), and use One Cycle Policy (with no annealation) to train the model for 24 epochs.`

&nbsp;

## Team - Weights_Heist
## Team Members - 

| Name        | mail           |
| ------------- |:-------------:|
|Gopinath Venkatesan|gops75[at]gmail[dot]com|
|Muhsin Mohammed|askmuhsin[at]gmail[dot]com|
|Rashu Tyagi|rashutyagi116[at]gmail[dot]com| 
|Subramanya R|subrananjun[at]gmail[dot]com| 

&nbsp;

## Model Description
The model architecture shown below details the sequence of layers / operations. One may note that the model uses two residual connections (Block 1 and Block 3) and rest are regular convolutions. Another observation could be that the MaxPool is located very near the output (one convolution away from the output Softmax layer, and yet the accuracy achieved is pretty good ~ 95%).

    https://github.com/eva7wandb/Eva7_Weights_Heist/blob/main/S9/README.md


### Model Architecture
![Screenshot](./resources/resnet9_architecture.png)

&nbsp;

### Model Summary
![Screenshot](./resources/eva7_S9_mdl_summary.png)

&nbsp;

## Learning Rate
Leslie Smith's [paper](https://arxiv.org/pdf/1506.01186.pdf) suggested a quick and an easy approach to tune the learning rate hyper-parameter while training, for a given dataset (here, CIFAR10) and model parameters, and rest of the hyper-parameters such as batch size and iterations.

To begin with, we conduct a learning rate range test, i.e. monotonously increasing learning rate in a given range with respect to each iteration of training samples of any specific fixed batch size. In our case, we set the initial learning rate to a very small value (1e-6) and with a set batch_size of 512 samples. 

An iterator is defined as an infinite loop to yield random samples of batch_size. For each iteration, the learning rate is increased in steps and the previous batch of images are replaced with the current batch of images, and the process is continued till the loss value for the classification criterion shoots up. 

The Losses vs Learning rate (Logarithmic) graph shows that the loss decreases first, reaches a tipping point and suddenly shoots up. We used David's [pytorch-lr-finder](https://github.com/davidtvs/pytorch-lr-finder) module to locate the best learning rate for further experimentation with the One Cycle Learning Rate Policy training of CIFAR10 samples.

&nbsp;

![Loss_vs_LearningRate](./resources/lr_vs_loss.png)

&nbsp;

## One Cycle Learning Rate (OCLR) Policy:

The one cycle policy consists of two steps of equal length, with the maximum learning rate chosen from the range test (described above) corresponding to the minimum loss. The lower or the starting learning rate is chosen as 1/5th to 1/10th of maximum learning rate or as the start of the descent of the learning rate. Now, the model training is performed starting with the lower learning rate, reaching the maximum learning rate during step 1, and then decreasing the learning rate till the starting learning rate is reached. Some variations can be tried along with the OCLR, like cutting short the second step a few iterations earlier, and reducing the learning rate from the endpoint (which is same as starting point value) further for the rest of the iterations -- referred to by the term 'Annealing'. Another variant could be, having the first half of short length, i.e. learning rate is ramped up quickly from starting LR to max LR (steep slope), and decreased at slow rates to sustain the larger updates to model parameters.

&nbsp;

## Training Logs

&nbsp;

## Misclassified Images / Grad-CAMs

Grad-CAMs can be implemented at the output layer of Block 2 where the feature map shape is 8 x 8 x 256. The misclassified images and their Grad-CAMs are shown below.



