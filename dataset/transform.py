import torchvision.transforms as transforms


def return_img_transform():
    transform = transforms.Compose(
        [
            # transforms.ToPILImage(),
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform
