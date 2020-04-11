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
device = 'cpu:0'


class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.output_size = output_size

        self.pre_model = nn.Sequential(nn.Linear(Z_DIM, 128 * 8 * 8))

        self.model = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 5, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 1, 5, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        z = self.pre_model(x)
        z = z.view(BATCH_SIZE, 128, 8, 8)
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Discriminator, self).__init__()

        def block(inputs, outputs, bn=True):
            c_block = [nn.Conv2d(inputs, outputs, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                c_block.append(nn.BatchNorm2d(outputs, 0.8))
            return c_block

        self.model = nn.Sequential(
            *block(1, 16, bn=False),
            *block(16, 32, bn=False),
            *block(32, 64, bn=False),
            *block(64, 128, bn=False)
        )

        self.last_layer = nn.Sequential(nn.Linear(512, 1))

    def forward(self, x):
        output = self.model(x)
        output = output.view(BATCH_SIZE, -1)
        output = self.last_layer(output)
        return output


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def gradient_penalty(discrim, real_inputs, fake_inputs):
    alpha = torch.randn((real_inputs.size(0), 1, 1, 1)).to(device)
    interpolates = ((alpha) * real_inputs + (1 - alpha) * fake_inputs).requires_grad_(True)
    d_interpolates = discrim(interpolates)
    target = Variable(torch.ones((real_inputs.size(0), 1)), requires_grad=False).to(device)
    gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates, grad_outputs=target, create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp


generator = Generator(Z_DIM, 256, X_SIZE).to(device)
discriminator = Discriminator(X_SIZE, 256).to(device)
# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)


d_optim = optim.Adam(discriminator.parameters(), lr=0.0002, betas=[0.5, 0.999])
g_optim = optim.Adam(generator.parameters(), lr=0.0002, betas=[0.5, 0.999])


picture_out_count = 0

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
        imgs = imgs.to(device)

        d_optim.zero_grad()

        # create fake inputs
        g_ = generator(Variable(torch.randn(BATCH_SIZE, Z_DIM)).to(device))

        #pass input through discriminator
        d_real_value = discriminator(imgs)
        d_real_error = -torch.mean(d_real_value)

        #pass fake through discriminator
        d_fake_value = discriminator(g_.detach())
        d_fake_error = torch.mean(d_fake_value)

        gp = gradient_penalty(discriminator, imgs, g_)

        d_total_error = d_real_error + d_fake_error + 10 * gp
        d_total_error.backward()
        d_optim.step()

        # train generator
        g_optim.zero_grad()
        g_ = generator(Variable(torch.randn(BATCH_SIZE, Z_DIM)).to(device))
        g_value = discriminator(g_)
        g_error = -torch.mean(g_value)
        g_error.backward()
        g_optim.step()


        if i % 5 == 0:
            samples = g_[:16].data.cpu().numpy() # get 16 pictures
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






