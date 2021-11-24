import numpy as np

import torch
from torch import nn


def kaiming_leaky_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')


def frequency_init(freq):
    def init(m):
        with torch.no_grad():
            if isinstance(m, nn.Linear):
                num_input = m.weight.size(-1)
                m.weight.uniform_(-np.sqrt(6 / num_input) / freq, np.sqrt(6 / num_input) / freq)
    return init


class MappingNetwork(nn.Module):
    def __init__(self, z_dim, hidden_dim, out_dim):
        super(MappingNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(hidden_dim, out_dim)
        )
        self.network.apply(kaiming_leaky_init)
        with torch.no_grad():
            self.network[-1].weight *= 0.25

    def forward(self, z):
        frequencies_offsets = self.network(z)
        frequencies = frequencies_offsets[..., :frequencies_offsets.shape[-1]//2]
        phase_shifts = frequencies_offsets[..., frequencies_offsets.shape[-1]//2:]
        return frequencies, phase_shifts


class FiLMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim)

    def forward(self, x, freq, phase_shift):
        x = self.layer(x)
        freq = freq.unsqueeze(1).expand_as(x)
        phase_shift = phase_shift.unsqueeze(1).expand_as(x)
        return torch.sin(freq * x + phase_shift)


class Generator(nn.Module):
    def __init__(self, input_dim=2, z_dim=64, hidden_dim=256, output_dim=3):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.network = nn.ModuleList([
            FiLMLayer(input_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
        ])
        self.rgb_layer = nn.Sequential(nn.Linear(hidden_dim, output_dim), nn.Sigmoid())

        self.mapping_network = MappingNetwork(z_dim, 256, len(self.network) * hidden_dim * 2)

        self.network.apply(frequency_init(25))
        self.rgb_layer.apply(frequency_init(25))

    def forward(self, inp, z):
        frequencies, phase_shifts = self.mapping_network(z)
        return self.forward_with_frequencies_phase_shifts(inp, frequencies, phase_shifts)

    def forward_with_frequencies_phase_shifts(self, inp, frequencies, phase_shifts):
        frequencies = frequencies*15 + 30

        x = inp
        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index+1) * self.hidden_dim
            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])
        rgb = self.rgb_layer(x)
        return rgb


if __name__ == '__main__':
    from utils import get_device, create_grid

    device = get_device()

    generator = Generator().to(device)

    z = torch.randn((16, 64)).to(device)
    grid = torch.randn((16, 25*34, 2)).to(device)

    out = generator(grid, z)
    out = out.view(16, 25, 34, 3)
