import argparse
import socket
import os
import urllib
import numpy as np
from PIL import Image
import random

from joblib import Parallel, delayed


def download_image(download_str, save_dir):
    img_name, img_url = download_str.strip().split('\t')
    save_img = os.path.join(save_dir, "{}.jpg".format(img_name))
    downloaded = False
    try:
        if not os.path.isfile(save_img):
            print("Downloading {} to {}.jpg".format(img_url, img_name))
            urllib.request.urlretrieve(img_url, save_img)

            # Check size of the images
            downloaded = True
            with Image.open(save_img) as img:
                width, height = img.size

            img_size_bytes = os.path.getsize(save_img)
            img_size_KB = img_size_bytes / 1024

            if width < 256 or height < 256 or img_size_KB < 10:
                os.remove(save_img)
                print("Remove downloaded images (w:{}, h:{}, s:{}KB)".format(
                    width, height, img_size_KB))
        else:
            print("Already downloaded {}".format(save_img))
    except Exception:
        if not downloaded:
            print("Cannot download.")
        else:
            print("Remove failed, downloaded images.")

        if os.path.isfile(save_img):
            os.remove(save_img)


def main():

    socket.setdefaulttimeout(10)
    
    with open(r"C:\Users\Sava\Documents\SRGAN\ImageNet\Images", encoding="utf8", errors='ignore') as f:
        lines = f.readlines()
        # lines = np.random.choice(lines, size=20000, replace=False)
        lines = [lines[i] for i in random.sample(range(0, len(lines)), 600000)]
    Parallel(n_jobs=12)(delayed(download_image)(
        line, 'D:\\Downloads\\ImageNet\\Images') for line in lines)


if __name__ == "__main__":
    main()
