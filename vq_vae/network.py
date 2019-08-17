import torch
import torch.nn as nn
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self, input_channels, residual_units):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=input_channels,
                      out_channels=residual_units,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=residual_units,
                      out_channels=input_channels,  # For residual
                      kernel_size=1, stride=1, bias=False)
        )

    def forward(self, inputs):
        return inputs + self._block(inputs)


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


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_units, num_layers, residual_hidden_Units):
        super(Encoder, self).__init__()
        self._1_conv = nn.Conv2d(in_channels=in_channels,
                                 out_channels=hidden_units//2,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._2_conv = nn.Conv2d(in_channels=hidden_units//2,
                                 out_channels=hidden_units,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._3_conv = nn.Conv2d(in_channels=hidden_units,
                                 out_channels=hidden_units,
                                 kernel_size=3,
                                 stride=2, padding=1)
        self.layers = nn.ModuleList()
        for i in range(len(num_layers)):
            self.layers.append(
                Residual(hidden_units, residual_hidden_Units)
            )

    def forward(self, inputs):
        x = self._1_conv(inputs)
        x = F.relu(x)

        x = self._2_conv(x)
        x = F.relu(x)

        x = self._3_conv(x)
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self, in_channels, hidden_units, num_layers, residual_hidden_units):
        super(Decoder, self).__init__()
        self._1_de_conv = nn.Conv2d(in_channels=in_channels,
                                    out_channels=hidden_units,
                                    kernel_size=3,
                                    stride=1, padding=1)
        self.residual_layers = nn.ModuleList()
        for i in range(len(num_layers)):
            self.residual_layers.append(
                Residual(hidden_units, residual_hidden_units)
            )

        self._2_de_trans_conv = nn.ConvTranspose2d(in_channels=hidden_units,
                                                   out_channels=hidden_units//2,
                                                   kernel_size=4,
                                                   stride=2, padding=1)
        self._3_de_trans_conv = nn.ConvTranspose2d(in_channels=hidden_units//2,
                                                   out_channels=3,
                                                   kernel_size=4,
                                                   stride=2, padding=1)

    def forward(self, inputs):
        x = self._1_de_conv(inputs)
        x = self.residual_layers(x)

        x = self._2_de_trans_conv(x)
        x = F.relu(x)

        return self._3_de_trans_conv(x)

