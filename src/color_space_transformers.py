import numpy as np
from sklearn.preprocessing import MinMaxScaler

def map_to_rgb(feature_map: np.ndarray):
    """ set to range 0..1 (RGB space) """
    return (feature_map - np.min(feature_map, axis=0)) / (np.max(feature_map, axis=0) - np.min(feature_map, axis=0))

def map_to_lab(feature_map: np.ndarray):
    """ this function does not work """
    # different ranges for channels L, a, b respectively
    ranges = [(0, 100), (-128, 127), (-128, 127)]

    H, W, C = feature_map.shape
    reshaped = feature_map.reshape(-1, C)

    normalized_channels = []
    for i, (lower, upper) in enumerate(ranges):
        scaler = MinMaxScaler(feature_range=(lower, upper))
        channel = scaler.fit_transform(reshaped[:, i].reshape(-1, 1))
        normalized_channels.append(channel)

    normalized_3d = np.hstack(normalized_channels).reshape(H, W, C)

    return normalized_3d
