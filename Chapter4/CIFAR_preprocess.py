from torch.utils.data import DataLoader as DL
import torchvision.transforms as Trans
import torchvision.datasets as ds
BATCH_SIZE = 5

# -----Commands to Download and Prepare the CIFAR10 Dataset-----

train_transform = Trans.Compose([
    Trans.RandomCrop(32, padding=4),
    Trans.RandomHorizontalFlip(),
    Trans.ToTensor(),
    Trans.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_transform = Trans.Compose([
    Trans.ToTensor(),
    Trans.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# train dataset
train_dataloader = DL(ds.CIFAR10('./data', train=True,
                                 download=True,
                                 transform=train_transform),
                      batch_size=BATCH_SIZE, shuffle=True)

# test dataset
test_dataloader = DL(ds.CIFAR10('./data', train=False,
                                transform=test_transform),
                     batch_size=BATCH_SIZE, shuffle=False)
print('CIFAR10 Dataset Pre-processing Done')


# -----Commands to Download and Prepare the CIFAR100 Dataset-----

train_transform = Trans.Compose([
    Trans.RandomCrop(32, padding=4),
    Trans.RandomHorizontalFlip(),
    Trans.ToTensor(),
    Trans.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_transform = Trans.Compose([
    Trans.ToTensor(),
    Trans.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# train dataset
train_dataloader = DL(ds.CIFAR100('./data', train=True,
                                  download=True,
                                  transform=train_transform),
                      batch_size=BATCH_SIZE, shuffle=True)

# test dataset
test_dataloader = DL(ds.CIFAR100('./data', train=False,
                                 transform=test_transform),
                     batch_size=BATCH_SIZE, shuffle=False)
print('CIFAR100 Dataset Pre-processing Done')