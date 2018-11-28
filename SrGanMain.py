from sklearn.feature_extraction import image
from SrGan import SrGan
import datetime
import random
import tensorflow as tf
import numpy as np
import cv2
from ImageLoader import ImageLoader
import os
import math

preload_epoch = 59
epoch = 1000
train = False
preload_model = True
pretrain=False
demo=True

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


if train:
    if preload_model:
        srgan = SrGan(epochs=epoch)
        srgan.load(epoch=preload_epoch, path='./experiment/')
    else:
        srgan = SrGan(epochs=epoch)
    srgan.train(preload_epoch=preload_epoch, initialize=not preload_model,
                validation_set_path="./ImageNet/TestImages", pretrain=pretrain)
    del srgan


if not train:
    srgan = SrGan(epochs=epoch)
    srgan.load(epoch=preload_epoch, path='./experiment/')
    if not demo:
        il = ImageLoader(
            batch_size=10, image_dir="./ImageNet/TestImages")
    else:
        il = ImageLoader(
            batch_size=1, image_dir="./ImageNet/DemoImages", shrink=False)
    for i, (input_images, target_images) in enumerate(il.getImages()):
        preds = srgan.predict(input_images)
        for i, img in enumerate(preds):
            pictures = [
                cv2.resize(input_images[i], dsize=(input_images[i].shape[1]*4, input_images[i].shape[0]*4),
                           interpolation=cv2.INTER_NEAREST),
                cv2.resize(input_images[i], dsize=(input_images[i].shape[1]*4, input_images[i].shape[0]*4),
                           interpolation=cv2.INTER_CUBIC),
                (img+1)/2, (target_images[i]+1)/2, ((img+1)/2 - (target_images[i]+1)/2)**2]

            pictures = (cv2.resize(img, dsize=(img.shape[1], img.shape[0]),
                                   interpolation=cv2.INTER_NEAREST) for img in pictures)

            numpy_horizontal = np.hstack(pictures)
            print(psnr((img+1)/2, (target_images[i]+1)/2))
            cv2.imshow('numpy_horizontal', numpy_horizontal)
            cv2.waitKey(0)
        cv2.destroyAllWindows()
