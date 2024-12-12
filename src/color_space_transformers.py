import numpy as np

def map_to_rgb(feature_map: np.ndarray):
    """ set to range 0..1 (RGB space) """
    return (feature_map - np.min(feature_map, axis=0)) / (np.max(feature_map, axis=0) - np.min(feature_map, axis=0))