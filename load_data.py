'''
Dataset loading and data transformation classes.
'''

from torch.utils.data import DataLoader
from torchvision import datasets, transforms




class FashionMNIST:

    def __init__(self, train_path, test_path, batch_size, shuffle, rotation_degrees=30, translate=(0,0.1), scale=(0.1, 0.2), shear=None):

        self.train_path = train_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rotation = rotation_degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear

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
                                                        shuffle=self.shuffle)

        test_loader = DataLoader(datasets.FashionMNIST(root=self.test_path,
                                                        train=False,
                                                        download=True,
                                                        transform=transforms.ToTensor()),
                                                        batch_size=self.batch_size,
                                                        shuffle=self.shuffle)


        return train_loader, test_loader

