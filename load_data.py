'''
Loads the given dataset using PyTorch Dataset module to be used with PyTorch's DataLoader Module for training/evaluation.
'''
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cfg
from utils import generate_training_data, image_rotations, image_shearing, image_flipping, image_noising


class Transform:
    '''
    Augments the image data (if enabled) and transforms the loaded arrays into torch tensors.
    '''

    def __init__(self, rotation_aug=cfg.ROTATION_ENABLED, shear_aug=cfg.SHEAR_ENABLED, flip_aug=cfg.FLIP_ENABLED, noise_aug=cfg.RANDOM_NOISE_ENABLED,
                 rotation_angles=cfg.ROTATION_ANGLES.copy(), shear_values=cfg.SHEAR_VALUES.copy(), flip_modes=cfg.FLIP_MODES.copy(),
                 noise_modes=cfg.NOISE_MODES.copy()):
        '''
        Initialize parameters related to image augmentations.
        '''
        self.rotation_augment = rotation_aug
        self.shear_augment = shear_aug
        self.flip_augment = flip_aug
        self.noise_augment = noise_aug
        self.rotation_angles = rotation_angles
        self.shear_values = shear_values
        self.flip_modes = flip_modes
        self.noise_modes = noise_modes


    def __call__(self, sample):
        '''
        Augments the image (if augment=True) and converts both the image and the label into Tensors.
        '''

        image, label = sample['image'], sample['label']

        #the image is copied so that the original image can be used for each augmentation process.
        #the original image is also reshaped to include batch size.
        augmented_images = image.copy().reshape((1, image.shape[0], image.shape[1], image.shape[2]))

        #image augmentation processes. After each augmentation, the augmented images will be concatenated (axis=0) into augmented_images array.
        #using np.vstacks is equivalent to concatenating the arrays in dimension 0.
        if self.rotation_augment:
            rotated_images = image_rotations(image=image, rotation_angles=self.rotation_angles)
            augmented_images = np.vstack((augmented_images, rotated_images)) #concatenate the augmented images together (incl. the original image)

        if self.shear_augment:
            sheared_images = image_shearing(image=image, shear_values=self.shear_values)
            augmented_images = np.vstack((augmented_images, sheared_images)) #concatenate the augmented images together (incl. the original image)

        if self.flip_augment:
            flipped_images = image_flipping(image=image, flip_modes=self.flip_modes)
            augmented_images = np.vstack((augmented_images, flipped_images)) #concatenate the augmented images together (incl. the original image)

        if self.noise_augment:
            noisy_images = image_noising(image=image, noise_modes=self.noise_modes)
            augmented_images = np.vstack((augmented_images, noisy_images)) #concatenate the augmented images together (incl. the original image)

        total_augmented_images = augmented_images.shape[0] #total images including the original images after augmentation process.

        label = np.tile(label, (total_augmented_images, 1)) #copy the same label across every augmented image data.

        #tranpose the channel dimension to front. PyTorch requires the dimension to be [channel, img_height, img_width] during training.
        augmented_images = np.transpose(augmented_images, [0, 3, 1, 2])

        #convert the augmented images (if any) into torch tensors and returns them.
        return {'image': torch.from_numpy(augmented_images), 'label': torch.from_numpy(label)}




class LoadDataset(Dataset):
    '''
    Contains overwritten methods from Torch's Dataset class.
    '''

    def __init__(self, resized_image_size=cfg.RESIZED_IMAGE_SIZE, total_images=cfg.TOTAL_DATA, classes=cfg.CLASSES, data_list=cfg.IMG_LABEL_LIST):
        '''
        Initiliaze dataset related parameters.
        '''
        self.resized_image_size = resized_image_size
        self.total_images = total_images
        self.classes = classes
        self.list_data = data_list
        self.transform = Transform()

    def __len__(self):
        '''
        Abstract method. Returns the total number of images.
        '''
        return self.total_images

    def __getitem__(self, idx):
        '''
        Abstract method. returns the image and label for a single input at index 'idx'.
        '''

        if torch.is_tensor(idx):
            idx = idx.tolist()

        image, label = generate_training_data(data=self.list_data[idx], resized_image_size=self.resized_image_size)

        sample = {
            'image':image,
            'label':label
        }

        sample = self.transform(sample)

        return sample


training = LoadDataset()
dataloader = DataLoader(training, batch_size=2, shuffle=True, num_workers=4)

for i, samp in enumerate(dataloader):

    print(samp['image'][0][0].shape)
    print(samp['image'][0].shape)
    print(samp['label'].shape)
    print(samp['label'])



    if i == 1:
        break