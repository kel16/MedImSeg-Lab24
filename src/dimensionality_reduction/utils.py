"""
Contains regular dimensionality reduction methods: PCA, MDS.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances

RANDOM_SEED = 42
COMPONENTS_COUNT = 3 # number of RGB components

def reduce_channels_mds(feature_map: np.ndarray):
    """
    Reduces arbitrary no. of channels to 3 using MDS.
    :param feature_map: Numpy array of shape (C, H, W)
    :return: image of shape (H, W, 3)
    """
    C, H, W = feature_map.shape
    feature_map_flat = feature_map.reshape(C, -1).T # Shape: (H*W, C)
    
    distance_matrix = pairwise_distances(feature_map_flat, metric='euclidean')
    embedding = MDS(n_components=COMPONENTS_COUNT, dissimilarity='precomputed', random_state=RANDOM_SEED)
    feature_transformed = embedding.fit_transform(distance_matrix)  # Shape: (H*W, 3)
    
    return feature_transformed.reshape(H, W, COMPONENTS_COUNT)

def reduce_channels_pca(feature_map):
    """
    Reduces arbitrary no. of channels to 3 using PCA.
    :param feature_map: Numpy array of shape (C, H, W)
    :return: image of shape (H, W, 3)
    """
    C, H, W = feature_map.shape
    feature_map_flat = feature_map.reshape(C, -1).T
    
    pca = PCA(n_components=COMPONENTS_COUNT, random_state=RANDOM_SEED)
    feature_transformed = pca.fit_transform(feature_map_flat)
    
    return feature_transformed.reshape(H, W, COMPONENTS_COUNT)
