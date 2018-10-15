import cv2
import os
import os.path
from operator import itemgetter
import random
from sklearn.feature_extraction import image
# print (cv2.__version__)


class ImageLoader:
    def __init__(self, batch_size, image_dir=r"C:\Users\Sava\Documents\SRGAN\ImageNet\TrainImages"):
        random.seed(123)
        self.batch_size = batch_size
        # specify your path here
        self.imageDir = image_dir
        # specify your vald extensions here
        self.valid_image_extensions = [
            ".jpg", ".jpeg", ".png", ".tif", ".tiff"]
        self.image_path_list = []
        for file in os.listdir(self.imageDir):
            extension = os.path.splitext(file)[1]
            if extension.lower() not in self.valid_image_extensions:
                continue
            self.image_path_list.append(os.path.join(self.imageDir, file))

    def shuffle_data(self):
        random.shuffle(self.image_path_list)

    def getImages(self):
        for i in range(len(self.image_path_list)//self.batch_size-1):
            images = [cv2.imread(
                imagePath) for imagePath in self.image_path_list[i*self.batch_size:(i+1)*self.batch_size]]
            yield self.preprocess_images(images)

    def preprocess_images(self, images):
        patch_images = [image.extract_patches_2d(image=img, patch_size=(
            96, 96), max_patches=16, random_state=1) for img in images if img is not None]

        target_images = [item for sublist in patch_images for item in sublist]
        input_images = [cv2.resize(
            img, dsize=(24, 24), interpolation=cv2.INTER_CUBIC) for img in target_images]
        target_images = [img/127.5-1 for img in target_images]
        input_images = [img/255 for img in input_images]
        return input_images, target_images
