import torch
import torchvision

# TODO: refactor this file
def get_dataset(task: str, batch_size: int, workers: int, shuffle_train=True, shuffle_test=False, return_dataset=False):
    if task in ['ffhq1024','ffhq1024-large']:
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        dataset = torchvision.datasets.ImageFolder('data/ffhq1024', transform=transforms)
        train_idx, test_idx = torch.arange(0, 60_000), torch.arange(60_000, len(dataset))
        train_dataset, test_dataset = torch.utils.data.Subset(dataset, train_idx), torch.utils.data.Subset(dataset, test_idx)
    elif task == 'ffhq256':
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        dataset = torchvision.datasets.ImageFolder('data/ffhq1024', transform=transforms)
        train_idx, test_idx = torch.arange(0, 60_000), torch.arange(60_000, len(dataset))
        train_dataset, test_dataset = torch.utils.data.Subset(dataset, train_idx), torch.utils.data.Subset(dataset, test_idx)
    elif task == 'ffhq128':
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        dataset = torchvision.datasets.ImageFolder('data/ffhq128', transform=transforms)
        train_idx, test_idx = torch.arange(0, 60_000), torch.arange(60_000, len(dataset))
        train_dataset, test_dataset = torch.utils.data.Subset(dataset, train_idx), torch.utils.data.Subset(dataset, test_idx)
    elif task == 'cifar10':
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        train_dataset = torchvision.datasets.CIFAR10('data', train=True, transform=transforms, download=True)
        test_dataset = torchvision.datasets.CIFAR10('data', train=False, transform=transforms, download=True)
    elif task == 'mnist':
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        train_dataset = torchvision.datasets.MNIST('data', train=True, transform=transforms, download=True)
        test_dataset = torchvision.datasets.MNIST('data', train=False, transform=transforms, download=True)
    elif task == 'kmnist':
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        train_dataset = torchvision.datasets.KMNIST('data', train=True, transform=transforms, download=True)
        test_dataset = torchvision.datasets.KMNIST('data', train=False, transform=transforms, download=True)
    else:
        print("> Unknown dataset. Terminating")
        exit()

    print(f"> Train dataset size: {len(train_dataset)}")
    print(f"> Test dataset size: {len(test_dataset)}")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=workers, shuffle=shuffle_train)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=workers, shuffle=shuffle_test)
    
    if return_dataset:
        return (train_loader, test_loader), (train_dataset, test_dataset)

    return train_loader, test_loader
