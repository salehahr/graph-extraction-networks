import glob
import os
from natsort import natsorted


def get_sorted_imgs(img_base_path: str, folder: str, small_dataset: bool = False) -> (str, str):
    paths = natsorted(glob.glob(os.path.join(img_base_path, f'**/{folder}/*.png'), recursive=True))

    if small_dataset:
        paths = paths[0:100]

    names = [os.path.basename(p) for p in paths]
    return paths, names
