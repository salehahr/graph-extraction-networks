import os
import random


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
