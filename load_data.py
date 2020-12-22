'''
Dataset loading and data transformation classes.
'''

from torch.utils.data import DataLoader
from torchvision import datasets, transforms




class FashionMNIST:

    def __init__(self, train_path, test_path, batch_size, shuffle, num_workers=4, rotation_degrees=30, translate=(0,0.1), scale=(0.1, 0.2), shear=None):

        self.train_path = train_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.rotation = rotation_degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.img_size = 28

    def __call__(self):

        train_loader = DataLoader(datasets.FashionMNIST(root=self.train_path,
                                                        train=True,
                                                        download=True,
                                                        transform=transforms.Compose([transforms.RandomAffine(
                                                                                                            degrees=self.rotation,
                                                                                                            translate=self.translate,
                                                                                                            scale=self.scale,
                                                                                                            shear=self.shear
                                                        ), transforms.ToTensor()])),
                                                        batch_size=self.batch_size,
                                                        shuffle=self.shuffle,
                                                        num_workers=self.num_workers)

        test_loader = DataLoader(datasets.FashionMNIST(root=self.test_path,
                                                        train=False,
                                                        download=True,
                                                        transform=transforms.ToTensor()),
                                                        batch_size=self.batch_size,
                                                        shuffle=self.shuffle,
                                                        num_workers=self.num_workers)


        return train_loader, test_loader, self.img_size

class Cifar10:

    def __init__(self, train_path, test_path, batch_size, shuffle, num_workers=4, rotation_degrees=30, translate=(0,0.1), scale=(0.1, 0.2), shear=None):

        self.train_path = train_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.rotation = rotation_degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.img_size = 32

    def __call__(self):

        train_loader = DataLoader(datasets.CIFAR10(root=self.train_path,
                                                        train=True,
                                                        download=True,
                                                        transform=transforms.Compose([transforms.RandomAffine(
                                                                                                            degrees=self.rotation,
                                                                                                            translate=self.translate,
                                                                                                            scale=self.scale,
                                                                                                            shear=self.shear
                                                        ), transforms.Grayscale(), transforms.ToTensor()])),
                                                        batch_size=self.batch_size,
                                                        shuffle=self.shuffle,
                                                        num_workers=self.num_workers)

        test_loader = DataLoader(datasets.CIFAR10(root=self.test_path,
                                                        train=False,
                                                        download=True,
                                                        transform=transforms.Compose([
                                                                                      transforms.Grayscale(),
                                                                                      transforms.ToTensor()])),
                                                        batch_size=self.batch_size,
                                                        shuffle=self.shuffle,
                                                        num_workers=self.num_workers)



        return train_loader, test_loader, self.img_size