from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_celeba_hq(data_dir='data/celeba-128'):
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1),
    ])
    return datasets.ImageFolder(data_dir, transform=transform)