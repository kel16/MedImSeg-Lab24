import matplotlib.pyplot as plt
from visualization_utils.color_space_transformers import map_to_lab

class LayerVisualizer():
    """
    utility to visualize a layer (original, reconstruction and latent)

    by default normalizes in LAB
    """
    def __init__(
        self,
        map_to_space = map_to_lab,
    ):
        self.map_to_space = map_to_space

    def _visualize(self, image):
        if image.shape[0] > 3:  # If more than 3 channels, average channels
            return image.mean(axis=0)
            # return image[0]
        else:
            return self.map_to_space(image)
        
    def plot(self, original, reconstructed, latent):
        """
        plots side-by-side original layer output, reconstruction by an autoencoder and compressed representation
        
        accepts images of shape (C, H, W)
        """
        # Create a side-by-side plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original Image
        axes[0].imshow(self._visualize(original), cmap='gray')
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        # Reconstructed Image
        axes[1].imshow(self._visualize(reconstructed), cmap='gray')
        axes[1].set_title("Reconstructed Image")
        axes[1].axis("off")

        # Latent Representation (3 channels, like RGB)
        axes[2].imshow(self._visualize(latent))
        axes[2].set_title("Latent Representation")
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()
