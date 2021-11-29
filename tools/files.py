import glob
import os
import random

from natsort import natsorted


def get_sorted_imgs(
    img_base_path: str, folder: str, small_dataset: bool = False
) -> (str, str):
    paths = natsorted(
        glob.glob(os.path.join(img_base_path, f"**/{folder}/*.png"), recursive=True)
    )

    if small_dataset:
        paths = paths[0:100]

    names = [os.path.basename(p) for p in paths]
    return paths, names


def get_random_video_path(base_path):
    generator = os.walk(base_path)
    path, subfolder_names = next(generator)[:2]

    if "raw" not in subfolder_names:
        # choose random video from subfolder
        video_id = random.randint(1, len(subfolder_names)) - 1
        video_path = os.path.join(path, subfolder_names[video_id])
        path = get_random_video_path(video_path)

    return path


def get_random_image(video_path):
    dir_raw = os.path.join(video_path, "cropped")

    generator = os.walk(dir_raw)
    img_list = next(generator)[-1]
    img_id = random.randint(1, len(img_list)) - 1

    return img_list[img_id]
