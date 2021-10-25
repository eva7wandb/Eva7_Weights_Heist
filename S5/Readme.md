# S5 - Coding Drill Down

In this session we go through many architectures starting from the very basic skeleton and try to achieve the goal of getting 99.4% accuracy on the MNIST dataset using under 10,000 parameters and 15 epochs.


# Round 0 

- get the code setup right.
- starting from the 8th iteration of class.
- not using any img aug, and lr steps now.
- the intention of this step is to have the correct code setup.

Architecture Round 0 - 

![Round 0 Architecture](https://github.com/eva7wandb/Eva7_Weights_Heist/blob/main/S5/Round%200%20Architecture.JPG)

Epochs logs : 

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
