#!/usr/bin/env python
# coding: utf-8

###############################################################################
# Link of our dataset:
# http://www2.isprs.org/commissions/comm3/wg4/semantic-labeling.html
##############################################################################


from PIL import Image
import numpy as np
import os.path
import argparse

parser = argparse.ArgumentParser('image processing')
parser.add_argument('--fold_A', dest='fold_A', help='input directory for images', type=str, default='./datasets/images/')
parser.add_argument('--fold_B', dest='fold_B', help='input directory for labels', type=str, default='./datasets/labels/')
parser.add_argument('--fold_AB', dest='fold_AB', help='output directory', type=str, default='./datasets/satellite/')
parser.add_argument('--num_imgs', dest='num_imgs', help='number of images', type=int, default=1000000)
args = parser.parse_args()


# define the funtion of combining image
def image_compose():
    to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE, IMAGE_ROW * IMAGE_SIZE)) #create a new image

    IMAGE_SAVE_PATH_TRAIN = os.path.join(IMAGE_SAVE_PATH, 'train')
    if not os.path.exists(IMAGE_SAVE_PATH_TRAIN):
        os.makedirs(IMAGE_SAVE_PATH_TRAIN)

    IMAGE_SAVE_PATH_TEST = os.path.join(IMAGE_SAVE_PATH, 'test')
    if not os.path.exists(IMAGE_SAVE_PATH_TEST):
        os.makedirs(IMAGE_SAVE_PATH_TEST)

    num_train = 0
    num_test = 0
    for name in image_names:
            from_image_a_row = Image.open(IMAGES_PATH_1 + name)
            from_image_b_row = Image.open(IMAGES_PATH_2 + name)

            from_image_width,from_image_height = from_image_a_row.size

            for i in range(int(from_image_width/IMAGE_SIZE)):
                for j in range(int(from_image_height / IMAGE_SIZE)):

                    from_image_a = from_image_a_row.crop((i*IMAGE_SIZE,j*IMAGE_SIZE,(i+1)*IMAGE_SIZE,(j+1)*IMAGE_SIZE))
                    from_image_b = from_image_b_row.crop((i*IMAGE_SIZE,j*IMAGE_SIZE,(i+1)*IMAGE_SIZE,(j+1)*IMAGE_SIZE))

                    to_image.paste(from_image_a, (0 * IMAGE_SIZE, 0 * IMAGE_SIZE))
                    to_image.paste(from_image_b, (1 * IMAGE_SIZE, 0 * IMAGE_SIZE))

                    if np.random.random() < 0.8:
                        to_image.save(os.path.join(IMAGE_SAVE_PATH_TRAIN, str(num_train+1)+'.jpg'),"JPEG") # save the new image as train sample
                        num_train = num_train + 1

                    else:

                        to_image.save(os.path.join(IMAGE_SAVE_PATH_TEST, str(num_test + 1) + '.jpg'), "JPEG")  # save the new image as test sample
                        num_test = num_test + 1
            
# convert the semantic image and ground_truth image to one paired image
# attention: the name of the paired semantic image and ground_truth image should be the same

# dataset of Vaihingen  
IMAGES_PATH_1 = args.fold_A  # ground truth immage, path should be changed
IMAGES_PATH_2 = args.fold_B  # semantic image , path should be changed
IMAGES_FORMAT = ['.tif']  
IMAGE_SIZE = 800
IMAGE_ROW = 1  
IMAGE_COLUMN = 2  
IMAGE_SAVE_PATH = args.fold_AB
 
# get all the name in the imageset
image_names = [name for name in os.listdir(IMAGES_PATH_2) for item in IMAGES_FORMAT if
               os.path.splitext(name)[1] == item]
 
image_compose() # call the function to get the aligned image





