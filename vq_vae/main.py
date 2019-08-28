import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam

import numpy as np

import matplotlib.pyplot as plt

from vq_vae.network import Model

training_data = CIFAR10(root='data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
                        ]))

training_variance = np.var(training_data.train_data / 255.0)

testing_data = CIFAR10(root='data', train=False, download=True,
                          transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
                          ]))


# configuration
batch_size = 32
hidden_units = 128
residual_layers = 2
residual_hidden_units = 32
num_embeddings = 512
embeddings_dim = 64
commitment_cost = 0.25
num_training_iterations = 25000
learning_rate = 3e-4


training_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
testing_loader = DataLoader(testing_data, batch_size=batch_size, shuffle=True)

model = Model(hidden_units, residual_layers, residual_hidden_units, num_embeddings, embeddings_dim, commitment_cost)
optimizer = Adam(model.parameters(), lr=learning_rate)

total_training_mean_error = []

for i in range(num_training_iterations):
    (data, _) = next(iter(training_loader))


    vq_loss, recon_data = model(data)
    recon_loss = torch.mean((recon_data - data) ** 2) / training_variance
    total_loss = vq_loss + recon_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    total_training_mean_error.append(total_loss.item())

    if (i + 1) % 100 == 0:
        print('%d iterations' % (i + 1))
        print('total_error: %.3f' % np.mean(total_training_mean_error[-100:]))
        print()

        image_data = data[0].numpy().transpose(2, 1, 0) + 0.5
        recon_image_data = recon_data.data[0].numpy().transpose(2, 1, 0) + 0.5
        plt.imshow(image_data)
        plt.savefig(f'img/data_{i}.png')
        plt.clf()
        plt.imshow(recon_image_data)
        plt.savefig(f'img/fakedata_{i}.png')
        plt.clf()