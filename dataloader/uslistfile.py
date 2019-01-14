import torch.utils.data as data


import os
import os.path

def dataloader(filepath):
    all_left_img, all_right_img = [os.path.join(filepath, 'TL{}.bmp'.format(i)) for i in range(10)], [os.path.join(filepath, 'TR{}.bmp'.format(i)) for i in range(10)]
    return all_left_img, all_right_img

if __name__ == '__main__':
    a, b = dataloader('Real')