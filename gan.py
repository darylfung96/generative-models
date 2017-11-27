import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import os

"""

"""

BATCH_SIZE = 32
Z_DIM = 100
X_SIZE = 784

class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size*2)
        self.fc3 = nn.Linear(hidden_size*2, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.sigmoid(self.fc3(x))

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Discriminator, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.sigmoid(self.fc3(x))


def sample_z():
    return torch.rand(BATCH_SIZE, Z_DIM)

mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

generator = Generator(Z_DIM, 100, X_SIZE)
discriminator = Discriminator(X_SIZE, 100)

d_optim = optim.Adam(discriminator.parameters(), lr=0.0001)
g_optim = optim.Adam(generator.parameters(), lr=0.0001)


picture_out_count = 0

for iteration in range(1000):

    for train_d_iter in range(100):
        d_optim.zero_grad()
        g_optim.zero_grad()

        x_, _ = mnist.train.next_batch(BATCH_SIZE)
        x_ = Variable(torch.from_numpy(x_))

        #pass input through discriminator
        d_real_value = discriminator(x_)
        d_real_error = F.binary_cross_entropy(d_real_value, Variable(torch.ones(BATCH_SIZE)))

        #pass fake through discriminator
        z_ = Variable(torch.rand(BATCH_SIZE, Z_DIM))
        g_ = generator(z_)
        d_fake_value = discriminator(g_)
        d_fake_error = F.binary_cross_entropy(d_fake_value, Variable(torch.zeros(BATCH_SIZE)))

        d_total_error = d_real_error + d_fake_error

        d_optim.step()

    for train_g_iter in range(100):
        g_optim.zero_grad()
        d_optim.zero_grad()

        g_ = generator(Variable(torch.rand(BATCH_SIZE, Z_DIM)))

        g_value = discriminator(g_)
        g_error = F.binary_cross_entropy(g_value, Variable(torch.ones(BATCH_SIZE)))
        g_error.backward()

        g_optim.step()


    if iteration % 10 == 0:
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






