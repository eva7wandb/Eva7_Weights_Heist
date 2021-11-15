#pip install --upgrade albumentations

import torch
import torchvision
import torchvision.transforms as transforms

import albumentations as A

import cv2
import numpy as np

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

class Cifar10Dataset(torchvision.datasets.CIFAR10):
    def __init__(self, root="./data", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label

def compute_mean_std(cifar10_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar10_training_dataset or cifar10_test_dataset
    
    Returns:
        a tuple contains mean, std value of entire dataset
    """
    print("Total number of images: ", len(cifar10_dataset))
    data_r = np.dstack([cifar10_dataset[i][0][:, :, 0] for i in range(len(cifar10_dataset))])
    data_g = np.dstack([cifar10_dataset[i][0][:, :, 1] for i in range(len(cifar10_dataset))])
    data_b = np.dstack([cifar10_dataset[i][0][:, :, 2] for i in range(len(cifar10_dataset))])
    mean = np.mean(data_r), np.mean(data_g), np.mean(data_b)
    std = np.std(data_r), np.std(data_g), np.std(data_b)

    return mean, std

mean, std = compute_mean_std(torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transforms.ToTensor()))
print("Mean = ", mean, "\n STD = ", std)


transform = A.Compose(
    [A.Normalize(mean, std),
     A.HorizontalFlip(),
     A.ShiftScaleRotate(),
     A.CoarseDropout(max_holes = 1, max_height=16, max_width=16, min_holes = 1, min_height=16, 
                     min_width=1, fill_value=mean, mask_fill_value = None),
     A.pytorch.transforms.ToTensorV2()
     ])

trainset = Cifar10Dataset(transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = Cifar10Dataset(train=False, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

