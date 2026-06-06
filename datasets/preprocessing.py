import numpy as np
import torch

def handle_nans(data_matrix, replacement_val=0.0):
    """
    Replaces NaN entries in numerical grids.
    """
    return np.nan_to_num(data_matrix, nan=replacement_val)

def crop_center(data_matrix, crop_h=512, crop_w=512):
    """
    Crops a central region of specified height and width from a 2D matrix.
    """
    h, w = data_matrix.shape
    start_h = h // 2 - crop_h // 2
    start_w = w // 2 - crop_w // 2
    return data_matrix[start_h : start_h + crop_h, start_w : start_w + crop_w]

def normalize_channels(image_tensor, method="minmax"):
    """
    Placeholders for data scale standardization.
    """
    # TODO: scientific validation required
    # Deferred from audit phase: input normalization needs physical limits definition
    return image_tensor
