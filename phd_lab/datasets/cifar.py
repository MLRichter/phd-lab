from typing import Tuple

from torch.utils.data import SubsetRandomSampler
from ..experiments.domain import DataBundle
from attr import attrs
from PIL import Image
import torch.utils.data
import numpy as np
from skimage.io import imshow, show

import torchvision
from torchvision import transforms

# Ulf: I am experiencing problems when setting num_workers > 0:
# the Dataloader simply freezes when trying to iterate over the data.
# This problem occurs with:
#    anaconda::pytorch-1.4.0-cuda101py38h0~
#    pytorch::pytorch-1.4.0-py3.8_cuda10.1.243_cudnn7.6.3_0
#
# The problem does not occur with:
#    pytorch::pytorch-1.6.0-py3.8_cuda10.1.243_cudnn7.6.3_0
#    
num_workers = 8

def Cifar10(batch_size=12, output_size=32, cache_dir='tmp') -> DataBundle:

    # Transformations
    RC = transforms.RandomCrop((32, 32), padding=4)
    RHF = transforms.RandomHorizontalFlip()
    NRM = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    TT = transforms.ToTensor()
    RS = transforms.Resize(output_size)

    # Transforms object for trainset with augmentation
    transform_with_aug = transforms.Compose([RC, RHF, RS, TT, NRM])
    # Transforms object for testset with NO augmentation
    transform_no_aug = transforms.Compose([RS, TT, NRM])


    trainset = torchvision.datasets.CIFAR10(root=cache_dir, train=True,
                                            download=True, transform=transform_with_aug)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers, pin_memory=True)
    testset = torchvision.datasets.CIFAR10(root=cache_dir, train=False,
                                           download=True, transform=transform_no_aug)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers, pin_memory=True)
    train_loader.name = "Cifar10"

    return DataBundle(
        dataset_name="Cifar10",
        train_dataset=train_loader,
        test_dataset=test_loader,
        cardinality=10,
        output_resolution=output_size,
        is_classifier=True
    )


@attrs(auto_attribs=True, slots=True)
class RandomPositioning(object):

    size: Tuple[int, int]
    random_noise: bool = False

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        img_arr = np.array(img)
        background = np.zeros((*self.size, 3), dtype=img_arr.dtype) if not self.random_noise else \
            np.random.uniform(img_arr.min(), img_arr.max(), (*self.size, 3)).astype(img_arr.dtype)
        start_x, start_y = np.random.randint(0, self.size[0]-img_arr.shape[0]), np.random.randint(0, self.size[1]-img_arr.shape[1])
        end_x, end_y = start_x + img_arr.shape[0], start_y + img_arr.shape[1]
        background[start_x:end_x, start_y:end_y, :] = img_arr
        #imshow(background)
        #show()
        return Image.fromarray(background)


    def __repr__(self):
        return self.__class__.__name__ + '(size={0}'.format(self.size)


def Cifar10SmallRandomPositioningNoise(batch_size=12, output_size=32, cache_dir='tmp') -> DataBundle:

    # Transformations
    RC = transforms.RandomCrop((32, 32), padding=4)
    RP = RandomPositioning((output_size, output_size), True)
    RHF = transforms.RandomHorizontalFlip()
    NRM = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    TT = transforms.ToTensor()
    RS = transforms.Resize(output_size)

    # Transforms object for trainset with augmentation
    transform_with_aug = transforms.Compose([RP, RHF, TT, NRM])
    # Transforms object for testset with NO augmentation
    transform_no_aug = transforms.Compose([RP, TT, NRM])


    trainset = torchvision.datasets.CIFAR10(root=cache_dir, train=True,
                                            download=True, transform=transform_with_aug)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers, pin_memory=True)
    testset = torchvision.datasets.CIFAR10(root=cache_dir, train=False,
                                           download=True, transform=transform_no_aug)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers, pin_memory=True)
    train_loader.name = "Cifar10SmallRandomPositioningNoise"

    return DataBundle(
        dataset_name="Cifar10SmallRandomPositioningNoise",
        train_dataset=train_loader,
        test_dataset=test_loader,
        cardinality=10,
        output_resolution=output_size,
        is_classifier=True
    )



def Cifar10SmallRandomPositioning(batch_size=12, output_size=32, cache_dir='tmp') -> DataBundle:

    # Transformations
    RC = transforms.RandomCrop((32, 32), padding=4)
    RP = RandomPositioning((output_size, output_size))
    RHF = transforms.RandomHorizontalFlip()
    NRM = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    TT = transforms.ToTensor()
    RS = transforms.Resize(output_size)

    # Transforms object for trainset with augmentation
    transform_with_aug = transforms.Compose([RP, RHF, TT, NRM])
    # Transforms object for testset with NO augmentation
    transform_no_aug = transforms.Compose([RP, TT, NRM])


    trainset = torchvision.datasets.CIFAR10(root=cache_dir, train=True,
                                            download=True, transform=transform_with_aug)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers, pin_memory=True)
    testset = torchvision.datasets.CIFAR10(root=cache_dir, train=False,
                                           download=True, transform=transform_no_aug)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers, pin_memory=True)
    train_loader.name = "Cifar10SmallRandomPositioning"

    return DataBundle(
        dataset_name="Cifar10SmallRandomPositioning",
        train_dataset=train_loader,
        test_dataset=test_loader,
        cardinality=10,
        output_resolution=output_size,
        is_classifier=True
    )


def ReflectBorderCifar10(batch_size=12, output_size=32, cache_dir='tmp') -> DataBundle:

    border = (output_size - 32) // 2
    uneven = output_size % 2

    # Transformations
    RC = transforms.RandomCrop((32, 32), padding=4)
    PAD = transforms.Pad((border, border, border + uneven, border + uneven), padding_mode='reflect')
    RHF = transforms.RandomHorizontalFlip()
    NRM = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    TT = transforms.ToTensor()
    RS = transforms.Resize(output_size)

    # Transforms object for trainset with augmentation
    transform_with_aug = transforms.Compose([RC, RHF, PAD, TT, NRM])
    # Transforms object for testset with NO augmentation
    transform_no_aug = transforms.Compose([PAD, TT, NRM])


    trainset = torchvision.datasets.CIFAR10(root=cache_dir, train=True,
                                            download=True, transform=transform_with_aug)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers, pin_memory=True)
    testset = torchvision.datasets.CIFAR10(root=cache_dir, train=False,
                                           download=True, transform=transform_no_aug)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers, pin_memory=True)
    train_loader.name = "ReflectBorderCifar10"

    return DataBundle(
        dataset_name="ReflectBorderCifar10",
        train_dataset=train_loader,
        test_dataset=test_loader,
        cardinality=10,
        output_resolution=output_size,
        is_classifier=True
    )


def EdgePaddingCifar10(batch_size=12, output_size=32, cache_dir='tmp') -> DataBundle:

    border = (output_size - 32) // 2
    uneven = output_size % 2

    # Transformations
    RC = transforms.RandomCrop((32, 32), padding=4, padding_mode="edge")
    PAD = transforms.Pad((border, border, border + uneven, border + uneven), padding_mode="edge")
    RHF = transforms.RandomHorizontalFlip()
    NRM = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    TT = transforms.ToTensor()
    RS = transforms.Resize(output_size)

    # Transforms object for trainset with augmentation
    transform_with_aug = transforms.Compose([RC, RHF, PAD, TT, NRM])
    # Transforms object for testset with NO augmentation
    transform_no_aug = transforms.Compose([PAD, TT, NRM])


    trainset = torchvision.datasets.CIFAR10(root=cache_dir, train=True,
                                            download=True, transform=transform_with_aug)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=8, pin_memory=True)
    testset = torchvision.datasets.CIFAR10(root=cache_dir, train=False,
                                           download=True, transform=transform_no_aug)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=8, pin_memory=True)
    train_loader.name = "EdgePaddingCifar10"

    return DataBundle(
        dataset_name="EdgePaddingCifar10",
        train_dataset=train_loader,
        test_dataset=test_loader,
        cardinality=10,
        output_resolution=output_size,
        is_classifier=True
    )


def BlackBorderCifar10(batch_size=12, output_size=32, cache_dir='tmp') -> DataBundle:

    border = (output_size - 32) // 2
    uneven = output_size % 2

    # Transformations
    RC = transforms.RandomCrop((32, 32), padding=4)
    PAD = transforms.Pad((border, border, border + uneven, border + uneven))
    RHF = transforms.RandomHorizontalFlip()
    NRM = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    TT = transforms.ToTensor()
    RS = transforms.Resize(output_size)

    # Transforms object for trainset with augmentation
    transform_with_aug = transforms.Compose([RC, RHF, PAD, TT, NRM])
    # Transforms object for testset with NO augmentation
    transform_no_aug = transforms.Compose([PAD, TT, NRM])


    trainset = torchvision.datasets.CIFAR10(root=cache_dir, train=True,
                                            download=True, transform=transform_with_aug)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers, pin_memory=True)
    testset = torchvision.datasets.CIFAR10(root=cache_dir, train=False,
                                           download=True, transform=transform_no_aug)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers, pin_memory=True)
    train_loader.name = "BlackBorderCifar10"

    return DataBundle(
        dataset_name="BlackBorderCifar10",
        train_dataset=train_loader,
        test_dataset=test_loader,
        cardinality=10,
        output_resolution=output_size,
        is_classifier=True
    )


def Cifar100(batch_size=12,
             output_size=(32,32),
             cache_dir='tmp') -> DataBundle:


    # Transformations
    RC = transforms.RandomCrop((32, 32), padding=4)
    RHF = transforms.RandomHorizontalFlip()
    NRM = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    TT = transforms.ToTensor()
    RS = transforms.Resize(output_size)

    # Transforms object for trainset with augmentation
    transform_with_aug = transforms.Compose([RC, RHF, RS, TT, NRM])
    # Transforms object for testset with NO augmentation
    transform_no_aug = transforms.Compose([RS, TT, NRM])

    trainset = torchvision.datasets.CIFAR100(root=cache_dir, train=True,
                                            download=True, transform=transform_with_aug)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)
    testset = torchvision.datasets.CIFAR100(root=cache_dir, train=False,
                                           download=True, transform=transform_no_aug)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)
    train_loader.name = "Cifar100"

    return DataBundle(
        dataset_name="Cifar100",
        train_dataset=train_loader,
        test_dataset=test_loader,
        cardinality=100,
        output_resolution=output_size,
        is_classifier=True
    )
