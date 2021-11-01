from torchvision import datasets, transforms
import torch

class MNIST:
    def __init__(
        self,
        cuda,
        path='./data',
        train_transforms=None,
        test_transforms=None,
    ):
        self.cuda = cuda
        if not train_transforms:
            self.train_transforms = transforms.Compose(
                [
                    transforms.RandomRotation((-7.0, 7.0), fill=(1,)),
                    #transforms.RandomAffine(degrees=(-7.0, 7.0), translate=(0.1,0.1), scale=(0.9, 1.1), fill=(1,)),
                    #transforms.ColorJitter(brightness=0.1, contrast=0.1),
                    #transforms.RandomAdjustSharpness(1.1),
                    #transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ]
            )
        else:
            self.train_transforms = train_transforms

        if not test_transforms:
            self.test_transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ]
            )
        else:
            self.test_transforms = test_transforms
            
        
        self.train = datasets.MNIST('/tmp/data', train=True, download=True, transform=self.train_transforms)
        self.test = datasets.MNIST('/tmp/data', train=False, download=True, transform=self.test_transforms)
            
        #data loader arguments - something you will fetch these from cmdt prompt


    def get_dataloader(self):
        train_dataloader_args = dict(
            shuffle=True, batch_size=64, num_workers=4, pin_memory=True
        ) if self.cuda else dict(shuffle=True, batch_size=64)

        test_dataloader_args = dict(
            shuffle=False, batch_size=64, num_workers=4, pin_memory=True
        ) if self.cuda else dict(shuffle=False, batch_size=64)
        
        train_dataloader = torch.utils.data.DataLoader(self.train, **train_dataloader_args)
        test_dataloader = torch.utils.data.DataLoader(self.test, **test_dataloader_args)
        
        return train_dataloader, test_dataloader

