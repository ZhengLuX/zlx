import torch
import numpy as np

def softmax_dim1(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, 1)

def Min_Max_normalization(image_array):
    return (image_array-np)

def Z_Score_normalization(image_array):
    return (image_array-np.mean(image_array)) / (np.std(image_array) + 1e-8)