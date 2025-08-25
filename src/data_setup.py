from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def create_dataloaders(data_dir: str, batch_size: int, num_workers: int):
    training_data = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]),
    )

    train_dataloader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    return train_dataloader
