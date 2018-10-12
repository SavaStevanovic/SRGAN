import cv2
import os, os.path

print (cv2.__version__)
def getImages(image_count=0):
    imageDir = r"C:\Users\Sava\Documents\SRGAN\ImageNet\Images" #specify your path here
    image_path_list = []
    valid_image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"] #specify your vald extensions here
    valid_image_extensions = [item.lower() for item in valid_image_extensions]

    for file in os.listdir(imageDir):
        extension = os.path.splitext(file)[1]
        if extension.lower() not in valid_image_extensions:
            continue
        image_path_list.append(os.path.join(imageDir, file))
        images=[]
    if image_count==0:
        images=[cv2.imread(imagePath) for imagePath in image_path_list]
        images=[image for image in images if image is not None]
        return images
    images=[cv2.imread(imagePath) for imagePath in image_path_list[:image_count]]
    images=[image for image in images if image is not None]
    return images