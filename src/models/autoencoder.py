import torch.nn as nn

class Autoencoder(nn.Module):

    def __init__(self, config=None):
        super().__init__()

        self.config = config

        self.encoder = nn.Sequential(
            nn.Linear(config['input_dim'], 4096),
            nn.Dropout(0.3),
            nn.GELU(),
            nn.BatchNorm1d(4096, affine=True),
            nn.Linear(4096, 2048),
            nn.GELU(),
            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.Dropout(0.3),
            nn.GELU(),
            #nn.BatchNorm1d(512, affine=True),
            nn.Linear(512, 256),
            nn.GELU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.GELU(),
            #nn.BatchNorm1d(512, affine=True),
            nn.Linear(512, 1024),
            nn.Dropout(0.3),
            nn.GELU(),
            nn.Linear(1024, 2048),
            nn.GELU(),
            nn.Linear(2048, 4096),
            nn.Dropout(0.3),
            nn.GELU(),
            nn.BatchNorm1d(4096, affine=True),
            nn.Linear(4096, config['input_dim'])
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def generate_embeddings(self, x):

        return self.encoder(x)