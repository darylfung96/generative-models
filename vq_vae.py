import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

    def forward(self, inputs):
        pass


class VectorQuantize(nn.Module):
    def __init__(self, num_embeddings, embeddings_dim, commitment_cost):
        super(VectorQuantize, self).__init__()
        self.num_embeddings = num_embeddings
        self.embeddings_dim = embeddings_dim
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(num_embeddings, embeddings_dim)
        self.embeddings.weight.data.uniform(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, inputs):
        flatten_inputs = inputs.view(-1, self.embeddings_dim)
        # find the nearest neighbour from the inputs to the embedding table
        distance = torch.sum(flatten_inputs**2, dim=1, keepdims=True) + \
                   torch.sum(self.embeddings.weight**2, dim=1) - \
                   2 * torch.matmul(flatten_inputs, self.embeddings.weight.t())

        selected_e_index = torch.argmin(distance, dim=1)
        one_hot_e_index = torch.zeros(selected_e_index.shape[0], self.num_embeddings)
        one_hot_e_index.scatter(1, selected_e_index, 1)

        e = torch.matmul(one_hot_e_index, self.embeddings.weight).view(inputs.shape)

        # find loss
        # e_loss (commitment loss is to control how much the latent produced from the encoder to
        # not deviate that much from the codebook)
        e_loss = torch.mean((e.detach() - inputs) ** 2)
        q_loss = torch.mean((inputs.detach() - e) ** 2)
        loss = self.commitment_cost * e_loss + q_loss

        # calculate the latent as the original z(output of encoder) in addition with the change from the
        # original z to the selected e (codebook)
        # this cause the latent to not diverge that much from the codebook
        latent = inputs + (e - inputs).detach()
        return loss, latent


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

    def forward(self, inputs):
        pass
