'''
Dataset loading and data transformation classes.
'''

from torch.utils.data import DataLoader
from torchvision import datasets, transforms




class FashionMNIST:

    def __init__(self, data_path, batch_size, shuffle, num_workers=4, rotation_degrees=30, translate=(0,0.2), scale=(0.95,1.2)):

        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.rotation = rotation_degrees
        self.translate = translate
        self.scale = scale
        self.img_size = 28
        self.num_class = 10

    def __call__(self):

        train_loader = DataLoader(datasets.FashionMNIST(root=self.data_path,
                                                        train=True,
                                                        download=True,
                                                        transform=transforms.Compose([transforms.RandomAffine(
                                                                                                            degrees=self.rotation,
                                                                                                            translate=self.translate,
                                                                                                            scale=self.scale
                                                        ), transforms.ToTensor()])),
                                                        batch_size=self.batch_size,
                                                        shuffle=self.shuffle,
                                                        num_workers=self.num_workers)

        test_loader = DataLoader(datasets.FashionMNIST(root=self.data_path,
                                                        train=False,
                                                        download=True,
                                                        transform=transforms.ToTensor()),
                                                        batch_size=self.batch_size,
                                                        shuffle=self.shuffle,
                                                        num_workers=self.num_workers)


        return train_loader, test_loader, self.img_size, self.num_class

class Cifar10:

    def __init__(self, data_path, batch_size, shuffle, num_workers=4, rotation_degrees=30, translate=(0,0.2), scale=(0.95, 1.2)):

        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.rotation = rotation_degrees
        self.translate = translate
        self.scale = scale
        self.img_size = 28
        self.num_class = 10

    def __call__(self):

        train_loader = DataLoader(datasets.CIFAR10(root=self.data_path,
                                                        train=True,
                                                        download=True,
                                                        transform=transforms.Compose([transforms.RandomAffine(
                                                                                                            degrees=self.rotation,
                                                                                                            translate=self.translate,
                                                                                                            scale=self.scale
                                                        ), transforms.Grayscale(), transforms.Resize(self.img_size), transforms.ToTensor()])),
                                                        batch_size=self.batch_size,
                                                        shuffle=self.shuffle,
                                                        num_workers=self.num_workers)

        test_loader = DataLoader(datasets.CIFAR10(root=self.data_path,
                                                        train=False,
                                                        download=True,
                                                        transform=transforms.Compose([
                                                                                      transforms.Grayscale(),
                                                                                      transforms.Resize(self.img_size),
                                                                                      transforms.ToTensor()])),
                                                        batch_size=self.batch_size,
                                                        shuffle=self.shuffle,
                                                        num_workers=self.num_workers)



        return train_loader, test_loader, self.img_size, self.num_class