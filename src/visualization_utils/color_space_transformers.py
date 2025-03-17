import numpy as np
from skimage import color

def map_to_rgb(feature_map: np.ndarray):
    """ normalize within range 0..1 (RGB space) """
    return (feature_map - np.min(feature_map, axis=0)) / (np.max(feature_map, axis=0) - np.min(feature_map, axis=0))


def normalize_channel(channel, target_low, target_high, percentile=5):
    # clip to handle outliers
    p_low = np.percentile(channel, percentile)
    p_high = np.percentile(channel, 100 - percentile)
    clipped = np.clip(channel, p_low, p_high)
    normalized = (clipped - p_low) / (p_high - p_low) * (target_high - target_low) + target_low
    
    return normalized

def map_to_lab(feature_map: np.ndarray):
    """
    :feature_map: of shape (channels, width, height)
    """
    L = normalize_channel(feature_map[0, ...], 0, 100)
    a = normalize_channel(feature_map[1, ...], -128, 127)
    b = normalize_channel(feature_map[2, ...], -128, 127)

    lab_image = np.stack([L, a, b], axis=-1)
    rgb_image = color.lab2rgb(lab_image)

    # clip values to valid RGB range
    return np.clip(rgb_image, 0, 1)
