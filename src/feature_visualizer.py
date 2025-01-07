import numpy as np
import matplotlib.pyplot as plt

DEFAULT_COLOR_MAP = 'viridis'
DEFAULT_FOLDER_NAME = './output'

class FeatureVisualizer():
    """
        visualizes a feature in a color space

        :param transform: function which fits a reduced feature into a certain color space
    """
    def __init__(self, transform, colormap=DEFAULT_COLOR_MAP, output_folder=DEFAULT_FOLDER_NAME):
        self.colormap = colormap
        self.output_folder = output_folder
        self.transform = transform
    
    def _get_image_path(self, file_name):
        return f'{self.output_folder}/{file_name}'
    
    def get_transform(self):
        return self.transform

    def plot_feature(self, feature_map, file_name=''):
        plt.imshow(self.transform(feature_map), cmap=self.colormap)
        plt.axis('off')

        if (file_name):
            plt.savefig(self._get_image_path(file_name))

        plt.show()
