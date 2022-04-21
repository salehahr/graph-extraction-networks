import os
import random
import re

pattern = r"(.*?)\/.*(\d{4}_\d{5}).png"


def shorten_filepath(filename: str, data_path) -> str:
    """Returns short version of the data filepath."""
    filename = os.path.relpath(filename, start=data_path)
    match = re.search(pattern, filename)
    vid_name, img_name = match.groups()
    return f"{vid_name}: {img_name}"


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
    dir_cropped = os.path.join(video_path, "cropped")

    generator = os.walk(dir_cropped)
    img_list = next(generator)[-1]
    img_id = random.randint(1, len(img_list)) - 1

    return img_list[img_id]


def create_folder(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)
