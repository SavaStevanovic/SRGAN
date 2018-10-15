from sklearn.feature_extraction import image
from SrGan import SrGan
import datetime
import random
import tensorflow as tf
import numpy as np
import cv2
from ImageLoader import ImageLoader

train = 0
epoch = 1

if (train == 1):
    srgan = SrGan(epochs=epoch)

    srgan.train()

    srgan.save(epoch=epoch)
    del srgan

    
if train==0:
    srgan = SrGan(epochs=epoch)
    srgan.load(epoch=epoch, path='./mse-model/')


    il = ImageLoader(
        batch_size=10, image_dir=r"C:\Users\Sava\Documents\SRGAN\ImageNet\TestImages")
    for i, (input_images, target_images) in enumerate(il.getImages(), 1):
        preds = srgan.predict(input_images)
        for i, img in enumerate(preds):
            pictures = [
                cv2.resize(input_images[i], dsize=(96, 96),
                        interpolation=cv2.INTER_NEAREST),
                cv2.resize(input_images[i], dsize=(96, 96),
                        interpolation=cv2.INTER_CUBIC),
                (img+1)/2, (target_images[i]+1)/2]

            pictures = (cv2.resize(img, dsize=(96*5, 96*5),
                                interpolation=cv2.INTER_NEAREST) for img in pictures)

            numpy_horizontal = np.hstack(pictures)
            cv2.imshow('numpy_horizontal', numpy_horizontal)
            cv2.waitKey(0)
        cv2.destroyAllWindows()
