import glob
import os
import re
import shutil

os.chdir("wandb")

files = glob.glob("**/**/model*")
pattern = r"run-\d*_\d*-(.*?)\\"

for file in files:
    run_id = re.match(pattern, file).group(1)
    new_fp = f"{run_id}.h5"

    shutil.copyfile(file, new_fp)
