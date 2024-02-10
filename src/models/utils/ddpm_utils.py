import torch
from torchvision import transforms
from matplotlib import pyplot as plt
import numpy as np

def get_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    return reverse_transforms(image)


def get_index_from_list(vals, t, x_shape,device):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu()).to(device)
    #out = vals.gather(-1, t.cpu).to(device)
    x = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    return x