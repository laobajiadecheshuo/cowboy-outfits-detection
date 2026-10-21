"""Download the 'cowboyoutfits' Kaggle competition dataset into the current folder.

Authenticates via the KAGGLE_API_TOKEN environment variable (the KGAT_... token
from kaggle.com/settings/api). Never hardcode the token in this file.
"""
import os
import shutil

import kagglehub

COMPETITION = "cowboyoutfits"
DEST_DIR = os.path.dirname(os.path.abspath(__file__))

# kagglehub downloads to a cache dir; we copy the contents into the current folder.
cache_path = kagglehub.competition_download(COMPETITION)
print("kagglehub cached at:", cache_path)

for entry in os.listdir(cache_path):
    src = os.path.join(cache_path, entry)
    dst = os.path.join(DEST_DIR, entry)
    if os.path.exists(dst):
        print("exists, skipping:", entry)
        continue
    if os.path.isdir(src):
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    print("downloaded:", entry)

print("\nDone. Files are in:", DEST_DIR)
