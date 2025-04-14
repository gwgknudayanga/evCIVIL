import torch
import torchvision
from torchvision import transforms

from typing import Tuple, List, Union, Type


class Dataset:
    def __init__(self, batch_sz: int=256, resize: Union[None, Tuple[int, int]]=None):
        self.batch_sz = batch_sz
        self.num_workers = 4
        self.trans = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]  # add here all the transforms you want
        if resize is not None:
            self.trans.insert(0, transforms.Resize(resize))
        
        self.train = None
        self.valid = None
    
    @property
    def n_classes(self) -> int:
        return len(self.labels)

    def get_dataloader(self, train) -> torch.utils.data.DataLoader:
        data = self.train if train else self.valid
        return torch.utils.data.DataLoader(data, batch_size=self.batch_sz, shuffle=train, num_workers=self.num_workers)
    
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return self.get_dataloader(True)
    
    def valid_dataloader(self) -> torch.utils.data.DataLoader:
        return self.get_dataloader(False)
    
    def text_labels(self, indices=List[int]) -> List[str]:
        raise NotImplementedError("You need to implement this method in the subclass")


class FashionMNIST(Dataset):
    def __init__(self, batch_sz: int=256, resize: Union[None, Tuple[int, int]]=None) -> None:
        super().__init__(batch_sz, resize)
        # self.trans.append(transforms.Grayscale(num_output_channels=3))  # example of how to add a transform
        trans = transforms.Compose(self.trans)
        self.train = torchvision.datasets.FashionMNIST(root="./data/", download=True, train=True, transform=trans)
        self.valid = torchvision.datasets.FashionMNIST(root="./data/", download=True, train=False, transform=trans)
        self.labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
        
    def text_labels(self, indices=List[int]) -> List[str]:
        return [self.labels[int(i)] for i in indices]


class CIFAR10(Dataset):
    def __init__(self, batch_sz: int=256, resize: Union[None, Tuple[int, int]]=None) -> None:
        super().__init__(batch_sz, resize)
        # self.trans.append(transforms.Grayscale(num_output_channels=3))  # example of how to add a transform
        trans = transforms.Compose(self.trans)
        self.train = torchvision.datasets.CIFAR10(root="./data/", download=True, train=True, transform=trans)
        self.valid = torchvision.datasets.CIFAR10(root="./data/", download=True, train=False, transform=trans)
        self.labels = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        
    def text_labels(self, indices=List[int]) -> List[str]:
        return [self.labels[int(i)] for i in indices]
    

class CIFAR100(Dataset):
    def __init__(self, batch_sz: int=256, resize: Union[None, Tuple[int, int]]=None) -> None:
        super().__init__(batch_sz, resize)
        # self.trans.append(transforms.Grayscale(num_output_channels=3))  # example of how to add a transform
        trans = transforms.Compose(self.trans)
        self.train = torchvision.datasets.CIFAR100(root="./data/", download=True, train=True, transform=trans)
        self.valid = torchvision.datasets.CIFAR100(root="./data/", download=True, train=False, transform=trans)
        self.labels = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
                  'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can',
                  'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud',
                  'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin',
                  'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo',
                  'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man',
                  'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange',
                  'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain',
                  'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road',
                  'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail',
                  'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
                  'tank', 'telephone', 'television','tiger','tractor','train','trout','tulip',
                  'turtle','wardrobe','whale','willow_tree','wolf','woman','worm']
        
    def text_labels(self, indices=List[int]) -> List[str]:
        return [self.labels[int(i)] for i in indices]
