import torch
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms


def risk_dataloador(
        Risk_Dataset,
        data_info,
        args,
        train=False,
        train_transform_method=False,
        RandomSampler_method=False
):

    # Data Aug
    # 1 Resized or Random Resized Crop (check)
    Resized = transforms.Resize(args.img_size)
    # RandomResizedCrop = transforms.RandomResizedCrop(args.img_size, scale=(0.5, 1.))
    # 2 Random Horizontal Flip
    RandomHorizontalFlip = transforms.RandomHorizontalFlip(p=0.5)
    # 3 Random Vertical Flip
    RandomVerticalFlip = transforms.RandomVerticalFlip(p=0.5)
    # 4 ColorJitter
    ColorJitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0)
    # # 5 Gaussian blur
    # Gaussianblur = GaussianBlur([.1, 2.])
    # # 6 Noise
    # Noise = transforms.Compose([
    #     ImgAugGaussianNoiseTransform(),
    #     lambda x: Image.fromarray(x)
    # ])
    # # 7 Gamma Correction
    # GammaCorrection = transforms.Compose([
    #     ImgAugGammaCorrectionTransform(),
    #     lambda x: Image.fromarray(x)
    # ])
    # # 8 Elastic
    # Elastic = transforms.Compose([
    #     ImgAugElasticTransform(),
    #     lambda x: Image.fromarray(x)
    # ])
    # 9 Random Rotation
    RandomRotation = transforms.RandomRotation(10)

    train_transform = transforms.Compose([
        Resized,
        RandomHorizontalFlip,
        RandomVerticalFlip,
        transforms.RandomApply([ColorJitter], p=0.5),  # important
        # # added more augument
        # transforms.RandomApply([Gaussianblur], p=0.5),
        # transforms.RandomApply([Noise], p=0.5),
        # transforms.RandomApply([GammaCorrection], p=0.5),
        # transforms.RandomApply([Elastic], p=0.5),
        #
        RandomRotation,
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
        # transforms.ToPILImage()
    ])

    test_transform = transforms.Compose([
        Resized,
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    if train_transform_method:
        transform = train_transform
    else:
        transform = test_transform

    if train:
        shuffle = True
        downsample = True
        batch_size = args.batch_size
    else:
        shuffle = False
        downsample = False
        # batch_size = 1
        batch_size = args.batch_size

    dataset = Risk_Dataset(
        args,
        data_info,
        # args.image_dir,
        transform,
        downsample)

    if RandomSampler_method:
        if args.label_bal == 'risk':
            class_sample_counts = np.bincount(dataset.risks)
            classcount = class_sample_counts.tolist()
            weights = 1. / torch.tensor(classcount, dtype=torch.float)
            sampleweights = weights[dataset.risks]
        elif args.label_bal == 'pos':
            class_sample_counts = np.bincount(dataset.labels)
            classcount = class_sample_counts.tolist()
            weights = 1. / torch.tensor(classcount, dtype=torch.float)
            sampleweights = weights[dataset.labels]
        else:
            raise ValueError(f" Balance the dataset by label: {args.label_bal} is not supported.")

        data_sampler = WeightedRandomSampler(sampleweights, len(dataset), replacement=True)
        data_loader = DataLoader(dataset=dataset, sampler=data_sampler, batch_size=batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True, drop_last=True)
    else:
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                                 num_workers=args.num_workers, pin_memory=True, drop_last=False)

    return data_loader
