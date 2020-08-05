'''
Configuration file.
'''
import os
import torch
from utils import get_classes, generate_img_label

####################################################### GENERAL CONFIGURATIONS #######################################################

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #identify the device to be used for training/evaluation/
DATASET_PATH = '/'.join(os.path.realpath(__file__).split('/')[:-1]) + '/dataset/' #absolute path of the dataset folder.
IMAGE_EXTS = ['jpg', 'png', 'jpeg', 'bmp']

MODEL_PATH = '/'.join(os.path.realpath(__file__).split('/')[:-1]) +'/torch_model/'

####################################################### IMAGE CONFIGURATIONS #######################################################

RESIZED_IMAGE_SIZE = 446

#IMAGE AUGMENTATION CONFIGURATIONS
ROTATION_ENABLED = True
SHEAR_ENABLED = True
FLIP_ENABLED = True
RANDOM_NOISE_ENABLED = True

ROTATION_ANGLES = [30, 45, 60, 90, 120]
SHEAR_VALUES = [-0.2, 0.2]
FLIP_MODES = ['lr', 'ud'] #Flips can be done either left-right or up-down.
NOISE_MODES = ['gaussian', 'salt', 'pepper', 's&p', 'speckle'] #skimage noise modes.

####################################################### DATASET CONFIGURATIONS #######################################################
CLASSES, NUM_CLASSES = get_classes(dataset_folder_path=DATASET_PATH) #get all the classes names and the number of classes.

IMG_LABEL_LIST = generate_img_label(dataset_path=DATASET_PATH, classes=CLASSES, img_exts=IMAGE_EXTS)

TOTAL_DATA = len(IMG_LABEL_LIST)

