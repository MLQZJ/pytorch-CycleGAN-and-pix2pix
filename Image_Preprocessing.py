#!/usr/bin/env python
# coding: utf-8

###############################################################################
# Link of our dataset:
# http://www2.isprs.org/commissions/comm3/wg4/semantic-labeling.html
##############################################################################


from PIL import Image
import os.path
import glob


# define the funtion of combining image
def image_compose():
    to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE, IMAGE_ROW * IMAGE_SIZE)) #create a new image
    
    for i,name in enumerate(image_names):
            from_image = Image.open(IMAGES_PATH_1 + name).resize(
                (IMAGE_SIZE, IMAGE_SIZE),Image.ANTIALIAS)
            to_image.paste(from_image, (0 * IMAGE_SIZE, 0 * IMAGE_SIZE))
            from_image = Image.open(IMAGES_PATH_2 + name).resize(
                (IMAGE_SIZE, IMAGE_SIZE),Image.ANTIALIAS)
            to_image.paste(from_image, (1 * IMAGE_SIZE, 0 * IMAGE_SIZE))
            to_image.save(os.path.join(IMAGE_SAVE_PATH, str(i+1)+'.jpg'),"JPEG") # save the new image
            
# convert the semantic image and ground_truth image to one paired image
# attention: the name of the paired semantic image and ground_truth image should be the same

# dataset of Vaihingen  
IMAGES_PATH_1 = "C:\\Users\\dell-pc\\Projets\\DeepLearning\\ISPRS_semantic_labeling_Vaihingen_Preprocessed\\"  # ground truth immage, path should be changed
IMAGES_PATH_2 = "C:\\Users\\dell-pc\\Projets\\DeepLearning\\ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE_Preprocessed\\" # semantic image , path should be changed
IMAGES_FORMAT = ['.tif']  
IMAGE_SIZE = 1024  
IMAGE_ROW = 1  
IMAGE_COLUMN = 2  
IMAGE_SAVE_PATH = "C:\\Users\\dell-pc\\Projets\\DeepLearning\\ISPRS_Vaihingen\\"  
 
# get all the name in the imageset
image_names = [name for name in os.listdir(IMAGES_PATH_2) for item in IMAGES_FORMAT if
               os.path.splitext(name)[1] == item]
 
image_compose() # call the function to get the aligned image


# dataset of Potsman
IMAGES_PATH_1 = "C:\\Users\\dell-pc\\Downloads\\2_Ortho_RGB\\"  # trainA
IMAGES_PATH_2 = "C:\\Users\\dell-pc\\Downloads\\5_Labels_all\\"  # trainB
IMAGES_FORMAT = ['.tif']  
IMAGE_SIZE = 1024  
IMAGE_ROW = 1  
IMAGE_COLUMN = 2  
IMAGE_SAVE_PATH = "C:\\Users\\dell-pc\\Projets\\DeepLearning\\ISPRS_Potsman\\"  

image_names = [name for name in os.listdir(IMAGES_PATH_2) for item in IMAGES_FORMAT if
               os.path.splitext(name)[1] == item]
image_compose() 





