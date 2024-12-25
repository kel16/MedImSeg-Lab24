import torch.nn as nn

KERNEL_SIZE = 3
STRIDE = 1
PADDING_SIZE = 1

class FeatureAutoencoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureAutoencoder, self).__init__()
        print(f"defined an autoencoder with input channels={in_channels} and out channels={out_channels}")

        # Encoder: Compress to 3 channels

        # self.encoder = nn.Sequential(
        #     nn.Conv2d(in_channels, out_channels, kernel_size=KERNEL_SIZE,
        #               stride=STRIDE, padding=PADDING_SIZE),
        #     nn.ReLU(),
        # )
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING_SIZE),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING_SIZE),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING_SIZE)  # Final 3-channel output
        )

        # Decoder: Reconstruct original channels

        # self.decoder = nn.Sequential(
        #     nn.Conv2d(out_channels, in_channels, kernel_size=KERNEL_SIZE,
        #               stride=STRIDE, padding=PADDING_SIZE),
        #     nn.ReLU(),
        # )
        self.decoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING_SIZE),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING_SIZE),
            nn.ReLU(),
            nn.Conv2d(64, in_channels, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING_SIZE)  # Back to original channels
        )
        
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)

        return latent, reconstructed
