import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.autograd import Variable
import random

import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

BATCH_SIZE = 32
LATENT_SIZE = 100


class CVariationalEncoder(nn.Module):

    def __init__(self, input_size, hidden_size, condition_size):
        super(CVariationalEncoder, self).__init__()
        self.hidden_size = hidden_size

        self.first_layer = nn.Linear(input_size + condition_size, hidden_size)
        # self.second_layer = nn.Linear(hidden_size, hidden_size)

        self.mean = nn.Linear(hidden_size, LATENT_SIZE)
        self.var = nn.Linear(hidden_size, LATENT_SIZE)

    def forward(self, input, condition):
        x = torch.cat([input, condition], 1)
        x = F.relu(self.first_layer(x))
        # x = F.relu(self.second_layer(x))

        mean = self.mean(x)
        log_var = self.var(x)

        return mean, log_var


class CVariationalDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, condition_size):
        super(CVariationalDecoder, self).__init__()
        self.hidden_size = hidden_size

        self.first_layer = nn.Linear(LATENT_SIZE + condition_size, hidden_size)
        # self.second_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, input, condition):
        x = torch.cat([input, condition], 1)
        x = F.relu(self.first_layer(x))
        # x = F.selu(self.second_layer(x))
        output = torch.sigmoid(self.output_layer(x))
        return output


def sample_dist(mean, log_var_sq):
    eps = Variable(torch.randn([BATCH_SIZE, LATENT_SIZE]))
    return mean + torch.exp(log_var_sq / 2) * eps


mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

encoder = CVariationalEncoder(784, 100, 10)
decoder = CVariationalDecoder(100, 784, 10)
params = list(encoder.parameters()) + list(decoder.parameters())
optimizer = Adam(params)
save_count = 0

for i in range(10000):
    data, target = mnist.train.next_batch(BATCH_SIZE)
    data = Variable(torch.from_numpy(data))
    c = Variable(torch.from_numpy(target.astype('float32')))

    mean, log_var_sq = encoder(data, c)
    z = sample_dist(mean, log_var_sq)
    output = decoder(z, c)

    reconstruction_loss = F.binary_cross_entropy(output, data, reduction='sum') / BATCH_SIZE
    kl_loss = torch.mean(torch.sum(-1. - log_var_sq + mean.pow(2) + log_var_sq.exp(), 1) / 2)

    loss = reconstruction_loss + kl_loss

    print(f"current iteration: {i}")
    print(f"current loss: {loss}\n")

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if i % 1000 == 0:
        c = np.zeros([BATCH_SIZE, 10], dtype='float32')
        number = random.randint(0, 9)
        c[:, number] = 1.

        c = Variable(torch.from_numpy(c))
        z = Variable(torch.randn([BATCH_SIZE, LATENT_SIZE]))
        output = decoder(z, c).data.numpy()[:16]

        fig = plt.figure(figsize=(4, 4))
        gs = GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)

        for index, s in enumerate(output):
            ax = plt.subplot(gs[index])
            plt.axis('off')
            ax.set_aspect('equal')
            plt.imshow(s.reshape(28, 28), cmap='Greys_r')

        if not os.path.exists('out/'):
            os.makedirs('out/')

        plt.savefig(f"out/{str(save_count).zfill(3)}_{number}.png")
        save_count += 1
        plt.close(fig)





