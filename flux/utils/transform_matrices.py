import numpy as np
import torch
import torch.nn.functional as F
import math

def identity():
    return torch.tensor([[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, 1]], dtype=torch.float32)

def translation(tx, ty):
    return torch.tensor([[1, 0, tx],
                         [0, 1, ty],
                         [0, 0, 1]], dtype=torch.float32)
    
def rotation(rad):
    return torch.tensor([[math.cos(rad), -math.sin(rad), 0],
                         [math.sin(rad), math.cos(rad), 0],
                         [0, 0, 1]], dtype=torch.float32)
    
def scale(sx, sy):
    return torch.tensor([[sx, 0, 0],
                         [0, sy, 0],
                         [0, 0, 1]], dtype=torch.float32)
    
def shear(shx, shy):
    return torch.tensor([[1, shx, 0],
                         [shy, 1, 0],
                         [0, 0, 1]], dtype=torch.float32)