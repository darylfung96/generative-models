import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as transforms

from torch.autograd import Variable
import torch.optim as optim

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import os

"""

"""

BATCH_SIZE = 512
Z_DIM = 100
X_SIZE = 784

class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.output_size = output_size

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size, 0.8),
            nn.LeakyReLU(0.2),

            nn.Linear(hidden_size, hidden_size*4),
            nn.BatchNorm1d(hidden_size*4, 0.8),
            nn.LeakyReLU(0.2),

            nn.Linear(hidden_size*4, output_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Discriminator, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size*4)
        self.fc2 = nn.Linear(hidden_size*4, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.2)
        return torch.sigmoid(self.fc3(x))



generator = Generator(Z_DIM, 256, X_SIZE)
discriminator = Discriminator(X_SIZE, 256)

d_optim = optim.Adam(discriminator.parameters(), lr=0.0002, betas=[0.5, 0.999])
g_optim = optim.Adam(generator.parameters(), lr=0.0002, betas=[0.5, 0.999])


picture_out_count = 0
BCELoss = nn.BCELoss()

dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "./data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)

for epoch in range(200):
    for i, (imgs, _) in enumerate(dataloader):
        d_optim.zero_grad()

        x_ = imgs.view(BATCH_SIZE, -1)

        # train generator
        g_optim.zero_grad()

        g_ = generator(Variable(torch.rand(BATCH_SIZE, Z_DIM)))
        g_value = discriminator(g_)
        g_error = BCELoss(g_value, Variable(torch.ones(BATCH_SIZE) * 0.9, requires_grad=False).unsqueeze(1))

        g_error.backward()
        g_optim.step()


        #pass input through discriminator
        d_real_value = discriminator(x_)
        d_real_error = BCELoss(d_real_value, Variable(torch.ones(BATCH_SIZE) * 0.9, requires_grad=False).unsqueeze(1))

        #pass fake through discriminator
        d_fake_value = discriminator(g_.detach())
        d_fake_error = BCELoss(d_fake_value, Variable(torch.zeros(BATCH_SIZE), requires_grad=False).unsqueeze(1))

        d_total_error = (d_real_error + d_fake_error)/2
        d_total_error.backward()
        d_optim.step()


        if i % 100 == 0:
            samples = g_[:16].data.numpy() # get 16 pictures
            fig = plt.figure(figsize=(4, 4))
            gs = gridspec.GridSpec(4, 4)
            gs.update(wspace=0.05, hspace=0.05)

            for i, sample in enumerate(samples):
                ax = plt.subplot(gs[i])
                plt.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')
                plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

            if not os.path.exists('out/'):
                os.makedirs('out/')

            plt.savefig('out/{}.png'.format(str(picture_out_count).zfill(3)))
            picture_out_count+=1
            plt.close(fig)

            print(f'epoch: {epoch} d_loss: {d_total_error} g_loss: {g_error}')






