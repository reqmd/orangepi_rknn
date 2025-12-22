from torchvision import transforms

def return_transforms(resolutions):
    train_transform = transforms.Compose([
        transforms.Resize((resolutions, resolutions)), 
        transforms.ToTensor(),
        #transforms.Normalize(mean=mean, std=std),
        #transforms.RandomGrayscale(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        #transforms.ColorJitter(0.1, 0.1, 0.1)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((resolutions, resolutions)), 
        #transforms.Normalize(mean=mean, std=std),
        transforms.ToTensor()
    ])
    return train_transform, val_transform