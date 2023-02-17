import os
import torch
import torchvision
import torchvision.transforms as transforms


def get_train_dataloader(args):
    if 'cifar' in args.dataset:
        args.train_image_size = 32
    transform_train = transforms.Compose(
        [transforms.RandomCrop(args.train_image_size), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])

    if 'cifar' in args.dataset:
        trainset = torchvision.datasets.CIFAR10(args.data_dir, train=True, transform=transform_train, download=True)
        args.image_size=32
    elif 'openimages' in args.dataset:
        trainset = torchvision.datasets.ImageFolder(os.path.join(args.data_dir, 'train'), transform=transform_train)
    else:
        raise NotImplementedError()

    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=3,
                                            drop_last=True,
                                            pin_memory=True, )
        
    return train_dataloader


def get_test_dataloader(args):
    if 'cifar' in args.dataset:
        args.image_size = 32
    transform_list = []
    if args.image_size > -1:
        transform_list += [transforms.Resize(args.image_size), transforms.CenterCrop(args.image_size)]
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    if 'cifar' in args.dataset:
        testset = torchvision.datasets.CIFAR10(args.data_dir, train=False, transform=transforms.Compose(transform_list), download=True)
    elif 'openimages' in args.dataset:
        testset = torchvision.datasets.ImageFolder(os.path.join(args.data_dir, 'val'), transform=transforms.Compose(transform_list))
    elif 'kodak' in args.dataset:
        testset = torchvision.datasets.ImageFolder(args.data_dir, transform=transforms.Compose(transform_list))
    else:
        raise NotImplementedError()
    test_dataloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                            shuffle=False, num_workers=1)
    return test_dataloader
     


