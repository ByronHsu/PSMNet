import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath):
    left_test = [os.path.join(filepath, 'TL{}.png'.format(i)) for i in range(10)]
    right_test = [os.path.join(filepath, 'TR{}.png'.format(i)) for i in range(10)]

    return left_test, right_test
