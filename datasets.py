from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from torch.nn import Module

from typing import Union

class CIFAR():

    def __init__(self, batch_size: int = 8, shuffle: bool = True, transform: Union[Module, None] = None, num_workers: int = 12):
        if transform == None:
            transform = transforms.Compose([
                ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        self.trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
        self.validset = CIFAR10(root='./data', train=False, download=True, transform=transform)
        self.trainloader = DataLoader(self.trainset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.validloader = DataLoader(self.validset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

class IN():

    def __init__(self, root: str = 'E:\Programming\Datasets\ImageNet', batch_size: int = 8, shuffle: bool = True, transform: Union[Module, None] = None, num_workers: int = 12):
        if transform == None:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop((224, 224)),
                ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        data = ImageFolder(root=root, transform=transform)

        self.trainset = Subset(data, list(range(10_000)))
        self.validset = Subset(data, list(range(10_000, 12_000)))
        self.testset = Subset(data, list(range(125_000, 175_000)))
        
        self.trainloader = DataLoader(self.trainset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.validloader = DataLoader(self.validset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.testloader = DataLoader(self.testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)