from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.nn import Module

class CIFAR():

    def __init__(self, batch_size: int = 8, shuffle: bool = True, transform: Module = ToTensor(), num_workers: int = 12):
        self.trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
        self.validset = CIFAR10(root='./data', train=False, download=True, transform=transform)
        self.trainloader = DataLoader(self.trainset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.validloader = DataLoader(self.validset, batch_size=batch_size, shuffle=False, num_workers=num_workers)