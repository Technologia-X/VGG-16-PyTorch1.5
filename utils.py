'''
Helper functions to be used across the program. This file does not import any other files as modules!
'''

import glob
from skimage import transform, util
from skimage.transform import rotate, AffineTransform, rescale
import numpy as np
import cv2

############################################################### GENERAL METHODS ###############################################################

def read_image(image_path, resized_image_size):
    '''
    Reads, resize, and normalize a single image from the given path and returns the numpy array of the image.
    '''

    im_ = cv2.imread(image_path)
    im_ = cv2.cvtColor(im_, cv2.COLOR_BGR2RGB)
    im_ = cv2.resize(im_, (resized_image_size, resized_image_size))
    im_ = im_/255 #normalize image

    return np.asarray(im_, dtype=np.float32)



############################################################### DATA RELATED METHODS ###############################################################
def get_classes(dataset_folder_path):
    '''
    Returns a list of class names (lowercased) from the dataset folder and the total number of classes.
    '''
    # class_names = []

    # for folder in glob.glob(dataset_folder_path + '**'):

    #     folder_name = folder.split('/')[-1].lower()
    #     class_names.append(folder_name)

    #This one-liner is equivalent to above commented lines of code.
    class_names = [folder.split('/')[-1].lower() for folder in glob.glob(dataset_folder_path + '**')]
    class_names.sort()

    return class_names, len(class_names)

def generate_img_label(dataset_path, classes, img_exts):
    '''
    Returns a list of tuples containing the absolute image path and the index of the class.
    '''

    # img_label = []

    # for path in glob.glob(dataset_path + '**', recursive=True):

    #     if path[-3:] in img_exts or path[-4:] in img_exts:

    #         class_name = path.split('/')[-2].lower()
    #         class_index = classes.index(class_name)

    #         img_label.append((path, class_index))

    #This one-liner is equivalent to above commented lines of code.
    img_label = [(path, classes.index(path.split('/')[-2].lower())) for path in glob.glob(dataset_path + '**', recursive=True)
                 if path[-3:] in img_exts or path[-4:] in img_exts]

    return img_label


def generate_training_data(data, resized_image_size):
    '''
    Reads the image path and label from the given data and returns them in numpy array.
    '''

    image_path, label = data

    image_array = read_image(image_path=image_path, resized_image_size=resized_image_size)

    return image_array, label


############################################################### IMAGE AUGMENTATION METHODS ###############################################################
def image_rotations(image, rotation_angles):
    '''
    Returns the rotated versions of the given image.
    '''
    rotated_images = [rotate(image, angle=x) for x in rotation_angles] #each rotated image will be stored in the list.
    return np.asarray(rotated_images, dtype=np.float32)

def image_shearing(image, shear_values):
    '''
    Returns the sheared versions of the given image.
    '''
    shearings = [AffineTransform(shear=x) for x in shear_values] #shearing objects

    #shear the image for every shear objects
    sheared_images = [transform.warp(image, shearings[i], order=1, preserve_range=True, mode='wrap') for i in range(len(shear_values))]

    return np.asarray(sheared_images, dtype=np.float32)

def image_flipping(image, flip_modes):
    '''
    Returns the flipped versions of the given image.
    '''
    flipped_images = []
    if 'ud' in flip_modes:
        flipped_images.append(np.flipud(image)) #flip the image up-down

    if 'lr' in flip_modes:
        flipped_images.append(np.fliplr(image)) #flip the image left-right

    return np.asarray(flipped_images, dtype=np.float32)

def image_noising(image, noise_modes):
    '''
    Returns the noisy version of the given image.
    '''
    noisy_images = [util.random_noise(image, mode=x) for x in noise_modes] #adds random noise to the original image based on the given mode(s).

    return np.asarray(noisy_images, dtype=np.float32)
