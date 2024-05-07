import torch.nn as nn

class Autoencoder(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.encoder = nn.Sequential(
            nn.Linear(config['input_size'], 128),
            nn.ReLU(),
        )