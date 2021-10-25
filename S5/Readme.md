# S5 - Coding Drill Down

In this session we go through many architectures starting from the very basic skeleton and try to achieve the goal of getting 99.4% accuracy on the MNIST dataset using under 10,000 parameters and 15 epochs.


# FINAL RESULT 

We achieved the best results in Round 3 where we required to use : 
- 5760 parameters
- 99.30% of training accuracy in 12th Epoch
- 99.40% of testing accuracy in 9th Epoch
- Best testing accuracy - 99.44% in 13th Epoch


# Round 0 

### Target -- To setup the basic code architecture before we start experimenting.

### Analysis --
- get the code setup right.
- starting from the 8th iteration of class.
- not using any img aug, and lr steps now.
- the intention of this step is to have the correct code setup.

## Architecture and Number of Parameters Round 0 - 

![Round 0 Architecture](https://github.com/eva7wandb/Eva7_Weights_Heist/blob/main/S5/Round%200%20Architecture.JPG)

## Epochs logs Round 0 : 

EPOCH: 0
Loss=0.24914208054542542 Batch_id=937 Accuracy=93.78: 100%|██████████| 938/938 [00:10<00:00, 88.39it/s] 

Test set: Average loss: 0.0621, Accuracy: 9799/10000 (97.99%)

EPOCH: 1
Loss=0.011888708919286728 Batch_id=937 Accuracy=97.90: 100%|██████████| 938/938 [00:10<00:00, 91.20it/s]

Test set: Average loss: 0.0303, Accuracy: 9910/10000 (99.10%)

EPOCH: 2
Loss=0.02023753523826599 Batch_id=937 Accuracy=98.32: 100%|██████████| 938/938 [00:10<00:00, 91.58it/s]

Test set: Average loss: 0.0275, Accuracy: 9914/10000 (99.14%)

EPOCH: 3
Loss=0.03516384959220886 Batch_id=937 Accuracy=98.41: 100%|██████████| 938/938 [00:10<00:00, 91.12it/s] 

Test set: Average loss: 0.0291, Accuracy: 9907/10000 (99.07%)

EPOCH: 4
Loss=0.15804673731327057 Batch_id=937 Accuracy=98.48: 100%|██████████| 938/938 [00:10<00:00, 90.60it/s]

Test set: Average loss: 0.0256, Accuracy: 9916/10000 (99.16%)

EPOCH: 5
Loss=0.011268964037299156 Batch_id=937 Accuracy=98.66: 100%|██████████| 938/938 [00:10<00:00, 91.30it/s]

Test set: Average loss: 0.0311, Accuracy: 9896/10000 (98.96%)

EPOCH: 6
Loss=0.04464658349752426 Batch_id=937 Accuracy=98.74: 100%|██████████| 938/938 [00:10<00:00, 90.32it/s]

Test set: Average loss: 0.0267, Accuracy: 9923/10000 (99.23%)

EPOCH: 7
Loss=0.003474261611700058 Batch_id=937 Accuracy=98.77: 100%|██████████| 938/938 [00:10<00:00, 89.49it/s]

Test set: Average loss: 0.0268, Accuracy: 9923/10000 (99.23%)

EPOCH: 8
Loss=0.05465521290898323 Batch_id=937 Accuracy=98.92: 100%|██████████| 938/938 [00:10<00:00, 91.37it/s] 

Test set: Average loss: 0.0223, Accuracy: 9936/10000 (99.36%)

EPOCH: 9
Loss=0.008698537945747375 Batch_id=937 Accuracy=99.03: 100%|██████████| 938/938 [00:10<00:00, 90.82it/s]

Test set: Average loss: 0.0220, Accuracy: 9937/10000 (99.37%)

EPOCH: 10
Loss=0.11097065359354019 Batch_id=937 Accuracy=99.07: 100%|██████████| 938/938 [00:10<00:00, 91.47it/s] 

Test set: Average loss: 0.0204, Accuracy: 9936/10000 (99.36%)

EPOCH: 11
Loss=0.0030628745444118977 Batch_id=937 Accuracy=99.02: 100%|██████████| 938/938 [00:10<00:00, 91.32it/s]

Test set: Average loss: 0.0199, Accuracy: 9936/10000 (99.36%)

EPOCH: 12
Loss=0.02697267197072506 Batch_id=937 Accuracy=99.10: 100%|██████████| 938/938 [00:10<00:00, 91.23it/s] 

Test set: Average loss: 0.0192, Accuracy: 9939/10000 (99.39%)

EPOCH: 13
Loss=0.0026952442713081837 Batch_id=937 Accuracy=99.08: 100%|██████████| 938/938 [00:10<00:00, 90.14it/s]

Test set: Average loss: 0.0199, Accuracy: 9939/10000 (99.39%)

EPOCH: 14
Loss=0.042014025151729584 Batch_id=937 Accuracy=99.11: 100%|██████████| 938/938 [00:10<00:00, 89.41it/s] 

Test set: Average loss: 0.0201, Accuracy: 9941/10000 (99.41%)


## Results 

Best Training Accuracy - 99.11% @15th Epoch
Best Test Accuracy - 99.41% @ 15th Epoch


# Round 1 

### Target -- As we are starting from Notebook 8 discussed in the session hence out goal here is to reduce the number of parameters by making changes in the architecture. 
          All the changes made follows the concepts covered in the 10 session notebooks. We came up with 7936 parameters after making changes.

### Analysis --
- studied the architectures covered in the session.
- starting from the 8th iteration of class.
- used some image augmentation like Random Rotation and as the data is not that complex hence we reduced the drop out rate to 0.05.
- made some changes in the architecture to reduce the number of parameters.
- Also learning rate was used as 0.08 (slightly larger than before so that loss could converge in the given number of epochs. 
- In LR Scheduler StepLR with step size = 8 was used.

## Architecture and Number of Parameters Round 1 - 

![Round 1 Architecture](https://github.com/eva7wandb/Eva7_Weights_Heist/blob/main/S5/Round%201%20Architecture.JPG)

## Epochs logs Round 1 : 

EPOCH: 0
Loss=0.021567178890109062 Batch_id=937 Accuracy=93.94: 100%|██████████| 938/938 [00:12<00:00, 75.69it/s]

Test set: Average loss: 0.0612, Accuracy: 9809/10000 (98.09%)

EPOCH: 1
Loss=0.02331358939409256 Batch_id=937 Accuracy=97.70: 100%|██████████| 938/938 [00:12<00:00, 75.90it/s]

Test set: Average loss: 0.0379, Accuracy: 9886/10000 (98.86%)

EPOCH: 2
Loss=0.08054167032241821 Batch_id=937 Accuracy=98.08: 100%|██████████| 938/938 [00:12<00:00, 75.42it/s]

Test set: Average loss: 0.0380, Accuracy: 9875/10000 (98.75%)

EPOCH: 3
Loss=0.041615795344114304 Batch_id=937 Accuracy=98.26: 100%|██████████| 938/938 [00:12<00:00, 75.96it/s]

Test set: Average loss: 0.0335, Accuracy: 9902/10000 (99.02%)

EPOCH: 4
Loss=0.021880824118852615 Batch_id=937 Accuracy=98.56: 100%|██████████| 938/938 [00:12<00:00, 76.55it/s]

Test set: Average loss: 0.0290, Accuracy: 9908/10000 (99.08%)

EPOCH: 5
Loss=0.029591504484415054 Batch_id=937 Accuracy=98.58: 100%|██████████| 938/938 [00:12<00:00, 76.65it/s]

Test set: Average loss: 0.0316, Accuracy: 9897/10000 (98.97%)

EPOCH: 6
Loss=0.006225215271115303 Batch_id=937 Accuracy=98.64: 100%|██████████| 938/938 [00:12<00:00, 76.42it/s]

Test set: Average loss: 0.0217, Accuracy: 9928/10000 (99.28%)

EPOCH: 7
Loss=0.006228962913155556 Batch_id=937 Accuracy=98.65: 100%|██████████| 938/938 [00:12<00:00, 76.17it/s]

Test set: Average loss: 0.0278, Accuracy: 9907/10000 (99.07%)

EPOCH: 8
Loss=0.1709238439798355 Batch_id=937 Accuracy=98.94: 100%|██████████| 938/938 [00:12<00:00, 76.42it/s]

Test set: Average loss: 0.0234, Accuracy: 9931/10000 (99.31%)

EPOCH: 9
Loss=0.004762503318488598 Batch_id=937 Accuracy=98.92: 100%|██████████| 938/938 [00:12<00:00, 76.87it/s]

Test set: Average loss: 0.0213, Accuracy: 9938/10000 (99.38%)

EPOCH: 10
Loss=0.010546272620558739 Batch_id=937 Accuracy=99.08: 100%|██████████| 938/938 [00:12<00:00, 75.51it/s]

Test set: Average loss: 0.0209, Accuracy: 9939/10000 (99.39%)

EPOCH: 11
Loss=0.0038530530873686075 Batch_id=937 Accuracy=99.08: 100%|██████████| 938/938 [00:12<00:00, 76.04it/s]

Test set: Average loss: 0.0202, Accuracy: 9942/10000 (99.42%)

EPOCH: 12
Loss=0.10972241312265396 Batch_id=937 Accuracy=99.05: 100%|██████████| 938/938 [00:12<00:00, 74.95it/s]

Test set: Average loss: 0.0202, Accuracy: 9944/10000 (99.44%)

EPOCH: 13
Loss=0.002599306171759963 Batch_id=937 Accuracy=99.14: 100%|██████████| 938/938 [00:12<00:00, 76.14it/s]

Test set: Average loss: 0.0198, Accuracy: 9946/10000 (99.46%)

EPOCH: 14
Loss=0.07659938186407089 Batch_id=937 Accuracy=99.04: 100%|██████████| 938/938 [00:12<00:00, 76.12it/s]

Test set: Average loss: 0.0194, Accuracy: 9945/10000 (99.45%)


## Results 

Best Training Accuracy - 99.14% @15th Epoch
Best Test Accuracy - 99.46% @ 13th Epoch


# Round 2 

### Target -- In the previous round we went down till 7936 parameters but this time we are making changes in the architecture such that this number could come down. It came down 
till 6616 number of parameters.

### Analysis --
- studied the architectures covered in the session.
- starting from the 8th iteration of class.
- used some image augmentation like Random Rotation and as the data is not that complex hence we reduced the drop out rate to 0.05.
- made some changes in the architecture to reduce the number of parameters. But the changes made were not beneficial for the accuracy Hence need to make the changes in the 
  model architecture more carefully.
- Also learning rate was used as 0.08 (slightly larger than before so that loss could converge in the given number of epochs. 
- In LR Scheduler StepLR with step size = 8 was used.

## Architecture and Number of Parameters Round 2 - 

![Round 2 Architecture](https://github.com/eva7wandb/Eva7_Weights_Heist/blob/main/S5/Round%202%20Architecture.JPG)

## Epochs logs Round 2 : 

EPOCH: 0
Loss=0.006810970604419708 Batch_id=937 Accuracy=94.05: 100%|██████████| 938/938 [00:14<00:00, 64.20it/s]

Test set: Average loss: 0.0563, Accuracy: 9820/10000 (98.20%)

EPOCH: 1
Loss=0.014805768616497517 Batch_id=937 Accuracy=97.72: 100%|██████████| 938/938 [00:14<00:00, 64.81it/s]

Test set: Average loss: 0.0376, Accuracy: 9882/10000 (98.82%)

EPOCH: 2
Loss=0.027132673189044 Batch_id=937 Accuracy=98.05: 100%|██████████| 938/938 [00:14<00:00, 64.90it/s]

Test set: Average loss: 0.0402, Accuracy: 9868/10000 (98.68%)

EPOCH: 3
Loss=0.09064525365829468 Batch_id=937 Accuracy=98.31: 100%|██████████| 938/938 [00:14<00:00, 64.17it/s]

Test set: Average loss: 0.0379, Accuracy: 9889/10000 (98.89%)

EPOCH: 4
Loss=0.06904435902833939 Batch_id=937 Accuracy=98.48: 100%|██████████| 938/938 [00:14<00:00, 64.68it/s]

Test set: Average loss: 0.0302, Accuracy: 9899/10000 (98.99%)

EPOCH: 5
Loss=0.024222832173109055 Batch_id=937 Accuracy=98.57: 100%|██████████| 938/938 [00:14<00:00, 64.46it/s]

Test set: Average loss: 0.0295, Accuracy: 9901/10000 (99.01%)

EPOCH: 6
Loss=0.002936724806204438 Batch_id=937 Accuracy=98.60: 100%|██████████| 938/938 [00:14<00:00, 65.03it/s]

Test set: Average loss: 0.0232, Accuracy: 9917/10000 (99.17%)

EPOCH: 7
Loss=0.036799706518650055 Batch_id=937 Accuracy=98.64: 100%|██████████| 938/938 [00:14<00:00, 65.20it/s]

Test set: Average loss: 0.0261, Accuracy: 9903/10000 (99.03%)

EPOCH: 8
Loss=0.25563138723373413 Batch_id=937 Accuracy=98.78: 100%|██████████| 938/938 [00:14<00:00, 64.49it/s]

Test set: Average loss: 0.0245, Accuracy: 9911/10000 (99.11%)

EPOCH: 9
Loss=0.00856387335807085 Batch_id=937 Accuracy=98.88: 100%|██████████| 938/938 [00:14<00:00, 65.16it/s]

Test set: Average loss: 0.0217, Accuracy: 9923/10000 (99.23%)

EPOCH: 10
Loss=0.05884627252817154 Batch_id=937 Accuracy=98.97: 100%|██████████| 938/938 [00:14<00:00, 64.20it/s]

Test set: Average loss: 0.0207, Accuracy: 9928/10000 (99.28%)

EPOCH: 11
Loss=0.0025583666283637285 Batch_id=937 Accuracy=99.00: 100%|██████████| 938/938 [00:14<00:00, 64.86it/s]

Test set: Average loss: 0.0205, Accuracy: 9930/10000 (99.30%)

EPOCH: 12
Loss=0.038112301379442215 Batch_id=937 Accuracy=98.99: 100%|██████████| 938/938 [00:14<00:00, 64.23it/s]

Test set: Average loss: 0.0212, Accuracy: 9929/10000 (99.29%)

EPOCH: 13
Loss=0.03332771360874176 Batch_id=937 Accuracy=99.00: 100%|██████████| 938/938 [00:14<00:00, 64.72it/s]

Test set: Average loss: 0.0198, Accuracy: 9934/10000 (99.34%)

EPOCH: 14
Loss=0.22595801949501038 Batch_id=937 Accuracy=99.08: 100%|██████████| 938/938 [00:14<00:00, 65.00it/s]

Test set: Average loss: 0.0196, Accuracy: 9934/10000 (99.34%)


## Results 

Best Training Accuracy - 99.08% @15th Epoch
Best Test Accuracy - 99.34% @ 14th Epoch


# Round 3 

### Target - In previous round the changes we made were not alligned with the accuracy we wanted in given number of epoch. Hence this time we are going more aggresive to make 
intelligent changes in the architecture by keeping number of channels low in initial layers. We dropped till 5760 parameters. 

### Analysis --
- studied the architectures covered in the session.
- starting from the 8th iteration of class.
- used some image augmentation like Random Rotation and as the data is not that complex hence we reduced the drop out rate to 0.01 so that we can get training accuracy 
  pushed furthur.
- in the round 2 we made some changes in the architecture to reduce the number of parameters. But the changes made were not beneficial for the accuracy Hence needed to make the   changes in the model architecture more carefully. So this time we reduced the number of parameters furthur to 5760 only. We reduced the number of channels expanded in the 
  initial layers. And kept it same for later layers.
- Also learning rate was increased furthur to  0.1 (larger than before so that loss could converge in the given number of epochs). 
- In LR Scheduler StepLR with step size = 8 was used.

## Architecture and Number of Parameters Round 3 - 

![Round 3 Architecture](https://github.com/eva7wandb/Eva7_Weights_Heist/blob/main/S5/Round%203%20Architecture.JPG)

## Epochs logs Round 3 : 

EPOCH: 0
Loss=0.37541574239730835 Batch_id=937 Accuracy=94.44: 100%|██████████| 938/938 [00:11<00:00, 79.09it/s]

Test set: Average loss: 0.0444, Accuracy: 9859/10000 (98.59%)

EPOCH: 1
Loss=0.07676538825035095 Batch_id=937 Accuracy=98.02: 100%|██████████| 938/938 [00:11<00:00, 79.20it/s]

Test set: Average loss: 0.0364, Accuracy: 9885/10000 (98.85%)

EPOCH: 2
Loss=0.017059624195098877 Batch_id=937 Accuracy=98.35: 100%|██████████| 938/938 [00:11<00:00, 78.21it/s]

Test set: Average loss: 0.0327, Accuracy: 9883/10000 (98.83%)

EPOCH: 3
Loss=0.17036381363868713 Batch_id=937 Accuracy=98.58: 100%|██████████| 938/938 [00:11<00:00, 78.38it/s]

Test set: Average loss: 0.0304, Accuracy: 9914/10000 (99.14%)

EPOCH: 4
Loss=0.004475714638829231 Batch_id=937 Accuracy=98.66: 100%|██████████| 938/938 [00:11<00:00, 79.49it/s]

Test set: Average loss: 0.0272, Accuracy: 9909/10000 (99.09%)

EPOCH: 5
Loss=0.004099765792489052 Batch_id=937 Accuracy=98.77: 100%|██████████| 938/938 [00:11<00:00, 79.04it/s]

Test set: Average loss: 0.0286, Accuracy: 9906/10000 (99.06%)

EPOCH: 6
Loss=0.008736259303987026 Batch_id=937 Accuracy=98.80: 100%|██████████| 938/938 [00:11<00:00, 79.25it/s]

Test set: Average loss: 0.0262, Accuracy: 9909/10000 (99.09%)

EPOCH: 7
Loss=0.01247859001159668 Batch_id=937 Accuracy=98.85: 100%|██████████| 938/938 [00:11<00:00, 79.84it/s]

Test set: Average loss: 0.0220, Accuracy: 9921/10000 (99.21%)

EPOCH: 8
Loss=0.0666138157248497 Batch_id=937 Accuracy=99.11: 100%|██████████| 938/938 [00:11<00:00, 80.01it/s]

Test set: Average loss: 0.0197, Accuracy: 9936/10000 (99.36%)

EPOCH: 9
Loss=0.046316105872392654 Batch_id=937 Accuracy=99.21: 100%|██████████| 938/938 [00:11<00:00, 80.41it/s]

Test set: Average loss: 0.0190, Accuracy: 9940/10000 (99.40%)

EPOCH: 10
Loss=0.015911150723695755 Batch_id=937 Accuracy=99.22: 100%|██████████| 938/938 [00:12<00:00, 78.07it/s]

Test set: Average loss: 0.0189, Accuracy: 9938/10000 (99.38%)

EPOCH: 11
Loss=0.007962384261190891 Batch_id=937 Accuracy=99.29: 100%|██████████| 938/938 [00:11<00:00, 80.11it/s]

Test set: Average loss: 0.0184, Accuracy: 9939/10000 (99.39%)

EPOCH: 12
Loss=0.0019018036546185613 Batch_id=937 Accuracy=99.30: 100%|██████████| 938/938 [00:11<00:00, 78.64it/s]

Test set: Average loss: 0.0182, Accuracy: 9938/10000 (99.38%)

EPOCH: 13
Loss=0.02076808735728264 Batch_id=937 Accuracy=99.29: 100%|██████████| 938/938 [00:11<00:00, 79.11it/s]

Test set: Average loss: 0.0184, Accuracy: 9944/10000 (99.44%)

EPOCH: 14
Loss=0.0008669004309922457 Batch_id=937 Accuracy=99.28: 100%|██████████| 938/938 [00:12<00:00, 77.75it/s]

Test set: Average loss: 0.0185, Accuracy: 9941/10000 (99.41%)


## Results 

Best Training Accuracy - 99.30% @12th Epoch
Best Test Accuracy - 99.40% @ 9th Epoch later received accuracy of 99.44% in 13th Epoch.
