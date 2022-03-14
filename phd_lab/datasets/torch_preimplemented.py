from os.path import exists, join
from typing import Tuple

from PIL import Image
from skimage.color import gray2rgb
from torch.utils.data import SubsetRandomSampler, DataLoader
from ..experiments.domain import DataBundle

import torch.utils.data
import numpy as np

import torchvision
from torchvision import transforms
from attr import attrs


def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight


def EuroSat(batch_size=12,
            output_size=64,
            cache_dir='tmp') -> DataBundle:

    RS = transforms.Resize(output_size)
    RC = transforms.RandomCrop(output_size, padding=output_size//8)
    RHF = transforms.RandomHorizontalFlip()
    NRM = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    TT = transforms.ToTensor()


    # Transforms object for trainset with augmentation
    transform_with_aug = transforms.Compose([RS, RHF, TT, NRM])
    # Transforms object for testset with NO augmentation
    transform_no_aug = transforms.Compose([RS, TT, NRM])

    train_dataset = torchvision.datasets.ImageFolder(root='../tmp/eurosat/train', transform=transform_with_aug)
    test_dataset = torchvision.datasets.ImageFolder(root='../tmp/eurosat/val', transform=transform_no_aug)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=4)

    num_classes = len(train_dataset.classes)

    return DataBundle(
        dataset_name="EuroSat",
        train_dataset=train_loader,
        test_dataset=test_loader,
        cardinality=num_classes,
        output_resolution=output_size,
        is_classifier=True
    )


def AgeFaceV3(batch_size=12,
            output_size=256,
            cache_dir='tmp') -> DataBundle:

    RS = transforms.Resize(output_size)
    RC = transforms.RandomCrop(output_size, padding=output_size//8)
    RHF = transforms.RandomHorizontalFlip()
    NRM = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    TT = transforms.ToTensor()


    # Transforms object for trainset with augmentation
    transform_with_aug = transforms.Compose([RS, RC, RHF, TT, NRM])
    # Transforms object for testset with NO augmentation
    transform_no_aug = transforms.Compose([RS, TT, NRM])

    train_dataset = torchvision.datasets.ImageFolder(root='./tmp/AgeFaceV3/train', transform=transform_with_aug)
    test_dataset = torchvision.datasets.ImageFolder(root='./tmp/AgeFaceV3/valid', transform=transform_no_aug)

    weights = make_weights_for_balanced_classes(train_dataset.imgs, len(train_dataset.classes))

    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, pin_memory=True, sampler=sampler)
    test_weights = make_weights_for_balanced_classes(test_dataset.imgs, len(test_dataset.classes))

    test_weights = torch.DoubleTensor(test_weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(test_weights, len(test_weights))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=4)
    return DataBundle(
        dataset_name="AgeFaceV3",
        train_dataset=train_loader,
        test_dataset=test_loader,
        cardinality=3,
        output_resolution=output_size,
        is_classifier=True
    )


def UrbanSounds(batch_size=12,
            output_size=256,
            cache_dir='tmp') -> DataBundle:

    RS = transforms.Resize(output_size)
    RC = transforms.RandomCrop(output_size, padding=output_size//8)
    RHF = transforms.RandomHorizontalFlip()
    NRM = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    TT = transforms.ToTensor()


    # Transforms object for trainset with augmentation
    transform_with_aug = transforms.Compose([RS, RC, RHF, TT, NRM])
    # Transforms object for testset with NO augmentation
    transform_no_aug = transforms.Compose([RS, TT, NRM])

    train_dataset = torchvision.datasets.ImageFolder(root=r'E:\urban_sounds\training_spec', transform=transform_with_aug)
    test_dataset = torchvision.datasets.ImageFolder(root=r'E:\urban_sounds\testing_spec', transform=transform_no_aug)

    weights = make_weights_for_balanced_classes(train_dataset.imgs, len(train_dataset.classes))

    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, pin_memory=True, sampler=sampler)
    test_weights = make_weights_for_balanced_classes(test_dataset.imgs, len(test_dataset.classes))

    test_weights = torch.DoubleTensor(test_weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(test_weights, len(test_weights))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=8)
    return DataBundle(
        dataset_name="UrbanSound",
        train_dataset=train_loader,
        test_dataset=test_loader,
        cardinality=10,
        output_resolution=output_size,
        is_classifier=True
    )


def AgeFaceV2(batch_size=12,
            output_size=256,
            cache_dir='tmp') -> DataBundle:

    RS = transforms.Resize(output_size)
    RC = transforms.RandomCrop(output_size, padding=output_size//8)
    RHF = transforms.RandomHorizontalFlip()
    NRM = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    TT = transforms.ToTensor()


    # Transforms object for trainset with augmentation
    transform_with_aug = transforms.Compose([RS, RC, RHF, TT, NRM])
    # Transforms object for testset with NO augmentation
    transform_no_aug = transforms.Compose([RS, TT, NRM])

    train_dataset = torchvision.datasets.ImageFolder(root='./tmp/AgeFaceV2/train', transform=transform_with_aug)
    test_dataset = torchvision.datasets.ImageFolder(root='./tmp/AgeFaceV2/valid', transform=transform_no_aug)

    weights = make_weights_for_balanced_classes(train_dataset.imgs, len(train_dataset.classes))

    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, pin_memory=True, sampler=sampler)
    test_weights = make_weights_for_balanced_classes(test_dataset.imgs, len(test_dataset.classes))

    test_weights = torch.DoubleTensor(test_weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(test_weights, len(test_weights))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=4)
    return DataBundle(
        dataset_name="AgeFaceV2",
        train_dataset=train_loader,
        test_dataset=test_loader,
        cardinality=4,
        output_resolution=output_size,
        is_classifier=True
    )


def AgeFace(batch_size=12,
            output_size=256,
            cache_dir='tmp') -> DataBundle:

    RS = transforms.Resize(output_size)
    RC = transforms.RandomCrop(output_size, padding=output_size//8)
    RHF = transforms.RandomHorizontalFlip()
    NRM = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    TT = transforms.ToTensor()


    # Transforms object for trainset with augmentation
    transform_with_aug = transforms.Compose([RS, RC, RHF, TT, NRM])
    # Transforms object for testset with NO augmentation
    transform_no_aug = transforms.Compose([RS, TT, NRM])

    train_dataset = torchvision.datasets.ImageFolder(root='./tmp/AgeFace/train', transform=transform_with_aug)
    test_dataset = torchvision.datasets.ImageFolder(root='./tmp/AgeFace/valid', transform=transform_no_aug)

    weights = make_weights_for_balanced_classes(train_dataset.imgs, len(train_dataset.classes))

    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, pin_memory=True, sampler=sampler)
    test_weights = make_weights_for_balanced_classes(test_dataset.imgs, len(test_dataset.classes))

    test_weights = torch.DoubleTensor(test_weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(test_weights, len(test_weights))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=4)
    return DataBundle(
        dataset_name="AgeFace",
        train_dataset=train_loader,
        test_dataset=test_loader,
        cardinality=4,
        output_resolution=output_size,
        is_classifier=True
    )


def Food101(batch_size=12,
            output_size=256,
            cache_dir='tmp') -> DataBundle:

    RS = transforms.Resize((output_size, output_size))
    RC = transforms.RandomCrop(output_size, padding=output_size//8)
    RHF = transforms.RandomHorizontalFlip()
    NRM = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    TT = transforms.ToTensor()


    # Transforms object for trainset with augmentation
    transform_with_aug = transforms.Compose([RS, RC, RHF, TT, NRM])
    # Transforms object for testset with NO augmentation
    transform_no_aug = transforms.Compose([RS, TT, NRM])

    train_dataset = torchvision.datasets.ImageFolder(root='./tmp/Food_101_Dataset/food-101/train', transform=transform_with_aug)
    test_dataset = torchvision.datasets.ImageFolder(root='./tmp/Food_101_Dataset/food-101/test', transform=transform_no_aug)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=4)
    return DataBundle(
        dataset_name="Food101",
        train_dataset=train_loader,
        test_dataset=test_loader,
        cardinality=101,
        output_resolution=output_size,
        is_classifier=True
    )


def ResizedImageNet16(batch_size=12, output_size=64, cache_dir="tmp") -> DataBundle:
    # Transformations
    from PIL import PngImagePlugin, ImageFile, Image
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    LARGE_ENOUGH_NUMBER = 1000000
    Image.MAX_IMAGE_PIXELS = 100000
    PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024 ** 3)
    RC = transforms.RandomCrop(16, padding=1)
    RHF = transforms.RandomHorizontalFlip()
    NRM = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    TT = transforms.ToTensor()
    RS = transforms.Resize(output_size)

    # Transforms object for trainset with augmentation
    transform_with_aug = transforms.Compose([RC, RHF, RS, TT, NRM])
    # Transforms object for testset with NO augmentation
    transform_no_aug = transforms.Compose([RS, TT, NRM])


    trainset = torchvision.datasets.ImageFolder(root='../tmp/ImageNetResized/train/lanczos', transform=transform_with_aug)
    testset = torchvision.datasets.ImageFolder(root='../tmp/ImageNetResized/val/lanczos', transform=transform_no_aug)


    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=3, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=True, num_workers=3, pin_memory=False)
    train_loader.name = "ResizedImageNet16"
    return DataBundle(
        train_dataset=train_loader,
        test_dataset=test_loader,
        cardinality=1000,
        output_resolution=output_size,
        is_classifier=True,
        dataset_name="ResizedImageNet16"
    )

def TinyImageNet(batch_size=12, output_size=64, cache_dir="tmp") -> DataBundle:
    # Transformations
    RC = transforms.RandomCrop(64, padding=64//8)
    RHF = transforms.RandomHorizontalFlip()
    NRM = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    TT = transforms.ToTensor()
    RS = transforms.Resize(output_size)

    # Transforms object for trainset with augmentation
    transform_with_aug = transforms.Compose([RC, RHF, RS, TT, NRM])
    # Transforms object for testset with NO augmentation
    transform_no_aug = transforms.Compose([RS, TT, NRM])


    trainset = torchvision.datasets.ImageFolder(root='./tmp/tiny-imagenet-200/train/', transform=transform_with_aug)
    testset = torchvision.datasets.ImageFolder(root='./tmp/tiny-imagenet-200/val/ds', transform=transform_no_aug)


    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=3, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=True, num_workers=3, pin_memory=False)
    train_loader.name = "TinyImageNet"
    return DataBundle(
        train_dataset=train_loader,
        test_dataset=test_loader,
        cardinality=200,
        output_resolution=output_size,
        is_classifier=True,
        dataset_name="TinyImageNet"
    )

def ResizedMalaria(batch_size=12, output_size=150, cache_dir="tmp") -> DataBundle:
    # Transformations
    #RC = transforms.RandomCrop(64, padding=64//8)
    RHF = transforms.RandomHorizontalFlip()
    RVF = transforms.RandomVerticalFlip()
    NRM = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    TT = transforms.ToTensor()
    RS = transforms.Resize((32, 32))
    RS2 = transforms.Resize((output_size, output_size))

    # Transforms object for trainset with augmentation
    transform_with_aug = transforms.Compose([RS, RS2, RVF, RHF, TT, NRM])
    # Transforms object for testset with NO augmentation
    transform_no_aug = transforms.Compose([RS, RS2, TT, NRM])


    trainset = torchvision.datasets.ImageFolder(root='./tmp/malaria/train/', transform=transform_with_aug)
    testset = torchvision.datasets.ImageFolder(root='./tmp/malaria/test/', transform=transform_no_aug)


    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=3, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=True, num_workers=3, pin_memory=False)
    train_loader.name = "ResizedMalaria"
    return DataBundle(
        train_dataset=train_loader,
        test_dataset=test_loader,
        cardinality=2,
        output_resolution=output_size,
        is_classifier=True,
        dataset_name="ResizedMalaria"
    )



def Malaria(batch_size=12, output_size=150, cache_dir="tmp") -> DataBundle:
    # Transformations
    #RC = transforms.RandomCrop(64, padding=64//8)
    RHF = transforms.RandomHorizontalFlip()
    RVF = transforms.RandomVerticalFlip()
    NRM = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    TT = transforms.ToTensor()
    RS = transforms.Resize((output_size, output_size))

    # Transforms object for trainset with augmentation
    transform_with_aug = transforms.Compose([RS, RVF, RHF, TT, NRM])
    # Transforms object for testset with NO augmentation
    transform_no_aug = transforms.Compose([RS, TT, NRM])


    trainset = torchvision.datasets.ImageFolder(root='./tmp/malaria/train/', transform=transform_with_aug)
    testset = torchvision.datasets.ImageFolder(root='./tmp/malaria/test/', transform=transform_no_aug)


    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=3, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=True, num_workers=3, pin_memory=False)
    train_loader.name = "Malaria"
    return DataBundle(
        train_dataset=train_loader,
        test_dataset=test_loader,
        cardinality=2,
        output_resolution=output_size,
        is_classifier=True,
        dataset_name="Malaria"
    )


def g2rgb(x):
    return x.repeat(3, 1, 1)


def AMNIST(batch_size=12, output_size=28, cache_dir='tmp') -> DataBundle:

    # Transformations
    #RC = transforms.RandomCrop(28, padding=3)
    #RHF = transforms.RandomHorizontalFlip()
    #NRM = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    TT = transforms.ToTensor()
    RS = transforms.Resize(output_size)
    RGB = transforms.Lambda(g2rgb)

    # Transforms object for trainset with augmentation
    transform_with_aug = transforms.Compose([RS, TT, RGB])
    # Transforms object for testset with NO augmentation
    transform_no_aug = transforms.Compose([RS, TT, RGB])


    trainset = torchvision.datasets.MNIST(root=cache_dir, train=True,
                                            download=True, transform=transform_with_aug)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=3, pin_memory=False)
    testset = torchvision.datasets.MNIST(root=cache_dir, train=False,
                                           download=True, transform=transform_no_aug)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=3, pin_memory=False)
    train_loader.name = "AMNIST"
    return DataBundle(
        dataset_name="AMNIST",
        train_dataset=train_loader,
        test_dataset=test_loader,
        cardinality=10,
        output_resolution=output_size,
        is_classifier=True
    )


def MNIST(batch_size=12, output_size=28, cache_dir='tmp') -> DataBundle:

    # Transformations
    RC = transforms.RandomCrop(28, padding=3)
    RHF = transforms.RandomHorizontalFlip()
    NRM = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    TT = transforms.ToTensor()
    RS = transforms.Resize(output_size)
    RGB = transforms.Lambda(g2rgb)

    # Transforms object for trainset with augmentation
    transform_with_aug = transforms.Compose([RC, RHF, RS, TT, RGB, NRM])
    # Transforms object for testset with NO augmentation
    transform_no_aug = transforms.Compose([RS, TT, RGB, NRM])


    trainset = torchvision.datasets.MNIST(root=cache_dir, train=True,
                                            download=True, transform=transform_with_aug)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=3, pin_memory=False)
    testset = torchvision.datasets.MNIST(root=cache_dir, train=False,
                                           download=True, transform=transform_no_aug)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=3, pin_memory=False)
    train_loader.name = "MNIST"
    return DataBundle(
        dataset_name="MNIST",
        train_dataset=train_loader,
        test_dataset=test_loader,
        cardinality=10,
        output_resolution=output_size,
        is_classifier=True
    )


# Lighting data augmentation take from here - https://github.com/eladhoffer/convNet.pytorch/blob/master/preprocess.py
class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()
        return img.add(rgb.view(3, 1, 1).expand_as(img))


def iNaturalist(batch_size=12, output_size=224, cache_dir='tmp') -> DataBundle:
    __imagenet_pca = {
        'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
        'eigvec': torch.Tensor([
            [-0.5675, 0.7192, 0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948, 0.4203],
        ])
    }


    train_tfms = transforms.Compose([
        transforms.Resize((output_size, output_size)),
        transforms.CenterCrop((output_size, output_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(.4,.4,.4),
        transforms.ToTensor(),
        Lighting(0.1, __imagenet_pca['eigval'], __imagenet_pca['eigvec']),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_tfms = transforms.Compose([
        transforms.Resize(int(output_size*1.14)),
        transforms.CenterCrop(output_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if exists("E:\\inaturalist\\"):
        path = "E:\\inaturalist\\"
    elif exists("./tmp/inaturalist"):
        path = "./tmp/inaturalist"
    else:
        raise ValueError("ImageNet not found")

    trainset = torchvision.datasets.ImageFolder(root=join(path, "train"), transform=train_tfms)
    testset = torchvision.datasets.ImageFolder(root=join(path, "test"), transform=val_tfms)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               shuffle=True, num_workers=6, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=6, pin_memory=True)
    train_loader.name = "iNaturalist"
    return DataBundle(
        dataset_name="iNaturalist",
        train_dataset=train_loader,
        test_dataset=test_loader,
        cardinality=1010,
        output_resolution=output_size,
        is_classifier=True
    )

def ResizediNaturalist(batch_size=12, output_size=224, cache_dir='tmp') -> DataBundle:
    __imagenet_pca = {
        'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
        'eigvec': torch.Tensor([
            [-0.5675, 0.7192, 0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948, 0.4203],
        ])
    }


    train_tfms = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Resize((output_size, output_size)),
        transforms.CenterCrop((output_size, output_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(.4,.4,.4),
        transforms.ToTensor(),
        Lighting(0.1, __imagenet_pca['eigval'], __imagenet_pca['eigvec']),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Resize(int(output_size*1.14)),
        transforms.CenterCrop(output_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if exists("E:\\inaturalist\\"):
        path = "E:\\inaturalist\\"
    elif exists("./tmp/inaturalist"):
        path = "./tmp/inaturalist"
    else:
        raise ValueError("ImageNet not found")

    trainset = torchvision.datasets.ImageFolder(root=join(path, "train"), transform=train_tfms)
    testset = torchvision.datasets.ImageFolder(root=join(path, "test"), transform=val_tfms)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               shuffle=True, num_workers=6, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=6, pin_memory=True)

    train_loader.name = "ResizediNaturalist"
    return DataBundle(
        dataset_name="ResizediNaturalist",
        train_dataset=train_loader,
        test_dataset=test_loader,
        cardinality=1010,
        output_resolution=output_size,
        is_classifier=True
    )



def ImageNette(batch_size=12, output_size=224, cache_dir='tmp') -> DataBundle:
    size = output_size
    __imagenet_pca = {
        'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
        'eigvec': torch.Tensor([
            [-0.5675, 0.7192, 0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948, 0.4203],
        ])
    }


    train_tfms = transforms.Compose([
        #transforms.Resize(output_size),
        #transforms.CenterCrop(output_size),
        transforms.RandomResizedCrop(output_size),
        transforms.RandomHorizontalFlip(),
        #transforms.ColorJitter(.4,.4,.4),
        transforms.ToTensor(),
        #Lighting(0.1, __imagenet_pca['eigval'], __imagenet_pca['eigvec']),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_tfms = transforms.Compose([
        transforms.Resize(int(output_size*1.14)),
        transforms.CenterCrop(output_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    if exists("../tmp/imagenette2"):
        path = "../tmp/imagenette2"
    elif exists("./tmp/imagenette2"):
        path = "./tmp/imagenette2"

    else:
        raise ValueError("imagenette2 not found at " + "./tmp/imagenette2")

    trainset = torchvision.datasets.ImageFolder(root=path+"/train", transform=train_tfms)
    testset = torchvision.datasets.ImageFolder(root=path+"/val", transform=val_tfms)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               shuffle=True, num_workers=6, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=6, pin_memory=True)
    train_loader.name = "ImageNette"
    return DataBundle(
        dataset_name="ImageNette",
        train_dataset=train_loader,
        test_dataset=test_loader,
        cardinality=10,
        output_resolution=output_size,
        is_classifier=True
    )


def ImageWoof(batch_size=12, output_size=224, cache_dir='tmp') -> DataBundle:
    size = output_size
    __imagenet_pca = {
        'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
        'eigvec': torch.Tensor([
            [-0.5675, 0.7192, 0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948, 0.4203],
        ])
    }


    train_tfms = transforms.Compose([
        #transforms.Resize(output_size),
        #transforms.CenterCrop(output_size),
        transforms.RandomResizedCrop(output_size),
        transforms.RandomHorizontalFlip(),
        #transforms.ColorJitter(.4,.4,.4),
        transforms.ToTensor(),
        #Lighting(0.1, __imagenet_pca['eigval'], __imagenet_pca['eigvec']),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_tfms = transforms.Compose([
        transforms.Resize(int(output_size*1.14)),
        transforms.CenterCrop(output_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    if exists("../tmp/imagewoof2"):
        path = "../tmp/imagewoof2"
    elif exists("./tmp/imagewoof2"):
        path = "./tmp/imagewoof2"
    else:
        raise ValueError("imagewoof2 not found")

    trainset = torchvision.datasets.ImageFolder(root=path+"/train", transform=train_tfms)
    testset = torchvision.datasets.ImageFolder(root=path+"/val", transform=val_tfms)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               shuffle=True, num_workers=6, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=6, pin_memory=True)
    train_loader.name = "ImageWoof"
    return DataBundle(
        dataset_name="ImageWoof",
        train_dataset=train_loader,
        test_dataset=test_loader,
        cardinality=10,
        output_resolution=output_size,
        is_classifier=True
    )


def ImageNet(batch_size=12, output_size=224, cache_dir='tmp') -> DataBundle:
    size = output_size
    __imagenet_pca = {
        'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
        'eigvec': torch.Tensor([
            [-0.5675, 0.7192, 0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948, 0.4203],
        ])
    }


    train_tfms = transforms.Compose([
        #transforms.Resize(output_size),
        #transforms.CenterCrop(output_size),
        transforms.RandomResizedCrop(output_size),
        transforms.RandomHorizontalFlip(),
        #transforms.ColorJitter(.4,.4,.4),
        transforms.ToTensor(),
        #Lighting(0.1, __imagenet_pca['eigval'], __imagenet_pca['eigvec']),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_tfms = transforms.Compose([
        transforms.Resize(int(output_size*1.14)),
        transforms.CenterCrop(output_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if exists("C:\\ImageNet"):
        path = "C:\\ImageNet"
    elif exists("./tmp/ImageNet"):
        path = "./tmp/ImageNet"
    else:
        raise ValueError("ImageNet not found")

    trainset = torchvision.datasets.ImageNet(root=path, transform=train_tfms, split='train')
    testset = torchvision.datasets.ImageNet(root=path, transform=val_tfms, split='val')

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               shuffle=True, num_workers=6, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=6, pin_memory=True)
    train_loader.name = "ImageNet"
    return DataBundle(
        dataset_name="ImageNet",
        train_dataset=train_loader,
        test_dataset=test_loader,
        cardinality=1000,
        output_resolution=output_size,
        is_classifier=True
    )


def ResizedImageNet(batch_size=12, output_size=224, cache_dir='tmp') -> DataBundle:
    size = output_size
    __imagenet_pca = {
        'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
        'eigvec': torch.Tensor([
            [-0.5675, 0.7192, 0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948, 0.4203],
        ])
    }

    train_tfms = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Resize((output_size, output_size)),
        transforms.CenterCrop((output_size, output_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(.4,.4,.4),
        transforms.ToTensor(),
        Lighting(0.1, __imagenet_pca['eigval'], __imagenet_pca['eigvec']),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Resize(int(output_size*1.14)),
        transforms.CenterCrop(output_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if exists("C:\\ImageNet"):
        path = "C:\\ImageNet"
    elif exists("./tmp/ImageNet"):
        path = "./tmp/ImageNet"
    else:
        raise ValueError("ImageNet not found")

    trainset = torchvision.datasets.ImageNet(root=path, transform=train_tfms, split='train')
    testset = torchvision.datasets.ImageNet(root=path, transform=val_tfms, split='val')

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               shuffle=True, num_workers=6, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=6, pin_memory=True)
    train_loader.name = "ResizedImageNet"
    return DataBundle(
        dataset_name="ResizedImageNet",
        train_dataset=train_loader,
        test_dataset=test_loader,
        cardinality=1000,
        output_resolution=output_size,
        is_classifier=True
    )


def TinyResizedFood101(batch_size=12, output_size=256,
                       cache_dir='tmp', shuffle_test: bool = False) -> DataBundle:
    RS = transforms.Resize(32)
    RS2 = transforms.Resize(output_size)
    RC = transforms.RandomCrop(output_size, padding=output_size//8)
    RHF = transforms.RandomHorizontalFlip()
    NRM = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    TT = transforms.ToTensor()

    # Transforms object for trainset with augmentation
    transform_with_aug = transforms.Compose([RS, RS2, RC, RHF, TT, NRM])
    # Transforms object for testset with NO augmentation
    transform_no_aug = transforms.Compose([RS, RS2, TT, NRM])

    train_dataset = torchvision.datasets.ImageFolder(root='./tmp/Food_101_Dataset/food-101/train', transform=transform_with_aug)
    test_dataset = torchvision.datasets.ImageFolder(root='./tmp/Food_101_Dataset/food-101/test', transform=transform_no_aug)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, shuffle=shuffle_test, num_workers=4)
    train_loader.name = "TinyResizedFood101"
    return DataBundle(
        dataset_name="TinyResizedFood101",
        train_dataset=train_loader,
        test_dataset=test_loader,
        cardinality=101,
        output_resolution=output_size,
        is_classifier=True
    )


@attrs(auto_attribs=True, slots=True)
class RandomPositioning(object):

    size: Tuple[int, int]

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        img_arr = np.array(img)
        if len(img_arr.shape) == 2:
            img_arr = gray2rgb(img_arr)
        background = np.zeros((*self.size, 3), dtype=img_arr.dtype)
        start_x, start_y = np.random.randint(0, self.size[0]-img_arr.shape[0]), np.random.randint(0, self.size[1]-img_arr.shape[1])
        end_x, end_y = start_x + img_arr.shape[0], start_y + img_arr.shape[1]
        background[start_x:end_x, start_y:end_y, :] = img_arr
        #imshow(background)
        #show()
        return Image.fromarray(background)


    def __repr__(self):
        return self.__class__.__name__ + '(size={0}'.format(self.size)


def MNISTSmallRandomPositioning(batch_size=12, output_size=32, cache_dir='tmp') -> DataBundle:

    # Transformations
    RC = transforms.RandomCrop((32, 32), padding=4)
    RP = RandomPositioning((output_size, output_size))
    RHF = transforms.RandomHorizontalFlip()
    NRM = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    TT = transforms.ToTensor()
    RS = transforms.Resize(output_size)
    RGB = transforms.Lambda(g2rgb)


    # Transforms object for trainset with augmentation
    transform_with_aug = transforms.Compose([RP, RHF, TT, NRM])
    # Transforms object for testset with NO augmentation
    transform_no_aug = transforms.Compose([RP, TT, NRM])


    trainset = torchvision.datasets.MNIST(root=cache_dir, train=True,
                                            download=True, transform=transform_with_aug)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=6, pin_memory=True)
    testset = torchvision.datasets.MNIST(root=cache_dir, train=False,
                                           download=True, transform=transform_no_aug)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=6, pin_memory=True)
    train_loader.name = "MNISTSmallRandomPositioning"

    return DataBundle(
        dataset_name="MNISTSmallRandomPositioning",
        train_dataset=train_loader,
        test_dataset=test_loader,
        cardinality=10,
        output_resolution=output_size,
        is_classifier=True
    )


def ZIEL_UC(batch_size=12, output_size=224, cache_dir='tmp') -> DataBundle:
    __imagenet_pca = {
        'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
        'eigvec': torch.Tensor([
            [-0.5675, 0.7192, 0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948, 0.4203],
        ])
    }


    train_tfms = transforms.Compose([
        transforms.Resize((output_size, output_size)),
        transforms.CenterCrop((output_size, output_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(.4,.4,.4),
        transforms.ToTensor(),
        Lighting(0.1, __imagenet_pca['eigval'], __imagenet_pca['eigvec']),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_tfms = transforms.Compose([
        transforms.Resize(int(output_size*1.14)),
        transforms.CenterCrop(output_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if exists("C:\\Users\\van30\\Documents\\yolov5\\datasets\\uncropped_dataset"):
        path = "C:\\Users\\van30\\Documents\\yolov5\\datasets\\uncropped_dataset"
    elif exists("./tmp/uncropped_dataset"):
        path = "./tmp/uncropped_dataset"
    else:
        raise ValueError("ZIEL-Dataset not found")

    trainset = torchvision.datasets.ImageFolder(root=join(path, "train"), transform=train_tfms)
    testset = torchvision.datasets.ImageFolder(root=join(path, "val"), transform=val_tfms)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               shuffle=True, num_workers=6, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=6, pin_memory=True)
    train_loader.name = "ZIEL_UC"
    return DataBundle(
        dataset_name="ZIEL_UC",
        train_dataset=train_loader,
        test_dataset=test_loader,
        cardinality=2,
        output_resolution=output_size,
        is_classifier=True
    )


def ZIEL_C(batch_size=12, output_size=224, cache_dir='tmp') -> DataBundle:
    __imagenet_pca = {
        'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
        'eigvec': torch.Tensor([
            [-0.5675, 0.7192, 0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948, 0.4203],
        ])
    }


    train_tfms = transforms.Compose([
        transforms.Resize((output_size, output_size)),
        transforms.CenterCrop((output_size, output_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(.4,.4,.4),
        transforms.ToTensor(),
        Lighting(0.1, __imagenet_pca['eigval'], __imagenet_pca['eigvec']),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_tfms = transforms.Compose([
        transforms.Resize(int(output_size*1.14)),
        transforms.CenterCrop(output_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if exists("C:\\Users\\van30\\Documents\\yolov5\\datasets\\cropped_dataset"):
        path = "C:\\Users\\van30\\Documents\\yolov5\\datasets\\cropped_dataset"
    elif exists("./tmp/cropped_dataset"):
        path = "./tmp/cropped_dataset"
    else:
        raise ValueError("ZIEL-Dataset not found")

    trainset = torchvision.datasets.ImageFolder(root=join(path, "train"), transform=train_tfms)
    testset = torchvision.datasets.ImageFolder(root=join(path, "val"), transform=val_tfms)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               shuffle=True, num_workers=6, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=6, pin_memory=True)
    train_loader.name = "ZIEL_C"
    return DataBundle(
        dataset_name="ZIEL_C",
        train_dataset=train_loader,
        test_dataset=test_loader,
        cardinality=2,
        output_resolution=output_size,
        is_classifier=True
    )

