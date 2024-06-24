"""
Code adopted from pix2pixHD:
https://github.com/NVIDIA/pix2pixHD/blob/master/data/image_folder.py
"""
import os

from PIL import Image

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def tensor2im(var):
    # var shape: (3, H, W)
    var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    var = ((var + 1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return Image.fromarray(var.astype('uint8'))


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), f'{dir} is not a valid directory'
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images

def make_dataset_from_paths_list(paths_file):
    assert os.path.exists(paths_file), f'{paths_file} is not a valid file'
    with open(paths_file, "r") as f:
        paths = f.readlines()
    paths = [p.strip() for p in paths]
    paths = [p for p in paths if is_image_file(p)]
    return paths
