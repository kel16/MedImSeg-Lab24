import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
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
        
    def plot(self, original, reconstructed, latent, title = '', plt_name = '', text = ''):
        """
        plots side-by-side original layer output, reconstruction by an autoencoder and compressed representation
        
        accepts images of shape (C, H, W)
        """
        # Create a side-by-side plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        if title:
            fig.suptitle(title)

        if text:
            plt.text(0.5, -0.2, text, ha="center", va="center", transform=plt.gca().transAxes)

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

        if plt_name:
            plot_name = f'./plots/plot-{plt_name}.png'
            print('Saving to ', plot_name)
            plt.savefig(plot_name)
        else:
            plt.show()

    def map_components(self, data, plt_name):
        rgb_data = data.reshape(data.shape[0], -1).T
        
        rgb_data_normalized = (rgb_data - np.min(rgb_data, axis=0)) / (
            np.max(rgb_data, axis=0) - np.min(rgb_data, axis=0)
        ) * 255
        rgb_colors = [f'rgb({r:.0f},{g:.0f},{b:.0f})' for r, g, b in rgb_data_normalized]

        # Create a 3D scatter plot using Plotly
        fig = go.Figure(data=[go.Scatter3d(
            x=rgb_data[:, 0],
            y=rgb_data[:, 1],
            z=rgb_data[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color=rgb_colors,
                opacity=0.8
            )
        )])

        # Update layout for better visualization
        fig.update_layout(
            scene=dict(
                xaxis=dict(title='R'),
                yaxis=dict(title='G'),
                zaxis=dict(title='B'),
            ),
            title=plt_name,
            margin=dict(l=0, r=0, b=0, t=40)
        )

        fig.write_html(f"./plots/map-{plt_name}.html")
        