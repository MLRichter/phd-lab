from torch.utils.data import SubsetRandomSampler
from ..experiments.domain import DataBundle

import torch.utils.data

import torchvision
from torchvision import transforms


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
                                              shuffle=True, num_workers=8, pin_memory=True)
    testset = torchvision.datasets.CIFAR10(root=cache_dir, train=False,
                                           download=True, transform=transform_no_aug)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=8, pin_memory=True)
    train_loader.name = "Cifar10"

    return DataBundle(
        dataset_name="Cifar10",
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
                                              shuffle=True, num_workers=8)
    testset = torchvision.datasets.CIFAR100(root=cache_dir, train=False,
                                           download=True, transform=transform_no_aug)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=8)
    train_loader.name = "Cifar100"

    return DataBundle(
        dataset_name="Cifar100",
        train_dataset=train_loader,
        test_dataset=test_loader,
        cardinality=100,
        output_resolution=output_size,
        is_classifier=True
    )
