import os
import random
import numpy as np

def get_image_paths(image_dir="./ImageNet/TrainImages"):
    # specify your path here
    imageDir = image_dir
    image_path_list = []
    for file in os.listdir(imageDir):
        image_path_list.append(file)
    return image_path_list


image_paths = np.array(get_image_paths())
test_paths = random.sample(range(len(image_paths)), 10000)
for img_path in image_paths[test_paths]:
    os.rename("./ImageNet/TrainImages/"+img_path,
              "./ImageNet/TestImages/"+img_path)
