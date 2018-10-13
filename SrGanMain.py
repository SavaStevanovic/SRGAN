from ImageLoader import getImages
from sklearn.feature_extraction import image
from SrGan import SrGan
import datetime
import random
import tensorflow as tf
import numpy as np
import cv2

# def get_image_samples(images, n_samples):
#     return [images[i] for i in random.sample(range(0, len(images)), n_samples)]

start_time = datetime.datetime.utcnow()

images = getImages(40)
print(datetime.datetime.utcnow()-start_time)

images = [image.extract_patches_2d(image=img, patch_size=(
    96, 96), max_patches=16, random_state=1) for img in images]
print(datetime.datetime.utcnow()-start_time)

target_images = [item/255 for sublist in images for item in sublist]
print(datetime.datetime.utcnow()-start_time)
input_images = [cv2.resize(
    img, dsize=(24, 24), interpolation=cv2.INTER_CUBIC) for img in target_images]

print(len(images))

srgan=SrGan(epochs=20)

srgan.train(training_set=(input_images,target_images))

