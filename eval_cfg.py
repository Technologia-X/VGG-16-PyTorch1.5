'''
Configuration file for model evaluation.
'''
import os
from utils import get_device, check_file_exist

CURR_FILE_PATH = '/'.join(os.path.realpath(__file__).split('/')[:-1]) #the absolute path of this file.
DEVICE = get_device() #identify the device to be used for training/evaluation
MODEL_PATH = CURR_FILE_PATH +'/torch_model/' #trained model save path.
MODEL_NAME = 'vgg16-torch.pth'

#checks if the torch model exists.
TRAINED_MODEL_PRESENCE = check_file_exist(file_path=MODEL_PATH, file_name=MODEL_NAME)


RESIZED_IMAGE_SIZE = 224

CLASS_FILE = './names.txt'
CLASSES = []

#Get the classes name from the .txt file.
OPEN_CFILE = open(CLASS_FILE, 'r')

#Reads every line in the file and append the name into the list.
for line in OPEN_CFILE:
    CLASSES.append(line.rstrip()) #strip the newline.

CLASSES.sort() #sort ascending order. IMPORTANT!
NUM_CLASSES = len(CLASSES)
