from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms

from torch.utils.data import DataLoader

training_data = CIFAR10(root='data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
                        ]))

testing_data = CIFAR10(root='data', train=False, download=True,
                          transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
                          ]))


# configuration
batch_size = 32

training_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
testing_loader = DataLoader(testing_data, batch_size=batch_size, shuffle=True)


