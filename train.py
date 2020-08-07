'''
Training script for VGG-16.
'''
from tqdm import tqdm
import torch
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from image_transforms import RandomRotate, RandomHorizontalFlip, RandomNoise, RandomVerticalFlip, RandomShear, ToTensor
from model import Model
from load_data import LoadDataset
from utils import calculate_accuracy, plot_graph
import train_cfg as t_cfg


########################################################## Model Initialization & Loading ##########################################################
VGG = Model(resized_img_size=t_cfg.RESIZED_IMAGE_SIZE, num_classes=t_cfg.NUM_CLASSES, init_weights=True)

OPTIMIZER = Adam(VGG.parameters(), lr=t_cfg.LEARNING_RATE) #optimizer
LR_DECAY = lr_scheduler.ExponentialLR(OPTIMIZER, gamma=t_cfg.LR_DECAY_RATE) #scheduler is used to lower the learning rate during training later.
LOSS_CRITERION = torch.nn.CrossEntropyLoss() #loss function.

VGG = VGG.to(t_cfg.DEVICE) #move the network to GPU if available.

print("--- Model Architecture ---")
print(VGG)

if t_cfg.TRAINED_MODEL_PRESENCE:

    MODEL_PARAMS = torch.load(t_cfg.MODEL_PATH+t_cfg.MODEL_NAME) #reads parameters from the model file.
    VGG.load_state_dict(MODEL_PARAMS) #load the parameters into the model architecture.
    print("Model parameters are loaded from the saved file!")



########################################################## Data Initialization & Loading ##########################################################
#Initialize the training data class.
TRAINING_DATA = LoadDataset(resized_image_size=t_cfg.RESIZED_IMAGE_SIZE, total_images=t_cfg.TOTAL_DATA, classes=t_cfg.CLASSES,
                            data_list=t_cfg.IMG_LABEL_LIST, transform=transforms.Compose([RandomRotate(angle_range=t_cfg.ROTATION_RANGE, prob=t_cfg.ROTATION_PROB),
                                                                                        RandomShear(shear_range=t_cfg.SHEAR_RANGE, prob=t_cfg.SHEAR_PROB),
                                                                                        RandomHorizontalFlip(prob=t_cfg.HFLIP_PROB),
                                                                                        RandomVerticalFlip(prob=t_cfg.VFLIP_PROB),
                                                                                        RandomNoise(mode=t_cfg.NOISE_MODE, prob=t_cfg.NOISE_PROB),
                                                                                        ToTensor(mode='training')]))

DATALOADER = DataLoader(TRAINING_DATA, batch_size=t_cfg.BATCH_SIZE, shuffle=t_cfg.DATA_SHUFFLE, num_workers=t_cfg.NUM_WORKERS)




########################################################## Model Training & Saving ##########################################################
BEST_ACCURACY = 0

ENTIRE_LOSS_LIST = []
ENTIRE_ACCURACY_LIST = []

for epoch_idx in range(t_cfg.EPOCH):

    print("Training for epoch %d has started!"%(epoch_idx+1))

    epoch_training_loss = []
    epoch_accuracy = []
    i = 0
    for i, sample in tqdm(enumerate(DATALOADER)):

        batch_x, batch_y = sample['image'].to(t_cfg.DEVICE), sample['label'].to(t_cfg.DEVICE)

        OPTIMIZER.zero_grad() #clear the gradients in the optimizer between every batch.

        net_output = VGG(batch_x) #output from the network.

        total_loss = LOSS_CRITERION(input=net_output, target=batch_y)

        epoch_training_loss.append(total_loss.item()) #append the loss of every batch.

        total_loss.backward() #calculate the gradients.
        OPTIMIZER.step()

        batch_acc = calculate_accuracy(network_output=net_output, target=batch_y)
        epoch_accuracy.append(batch_acc.cpu().numpy())

    LR_DECAY.step() #decay rate update
    curr_accuracy = sum(epoch_accuracy)/i
    curr_loss = sum(epoch_training_loss)

    print("The accuracy at epoch %d is %g"%(epoch_idx, curr_accuracy))
    print("The loss at epoch %d is %g"%(epoch_idx, curr_loss))

    ENTIRE_ACCURACY_LIST.append(curr_accuracy)
    ENTIRE_LOSS_LIST.append(curr_loss)


    if curr_accuracy > BEST_ACCURACY:

        torch.save(VGG.state_dict(), t_cfg.MODEL_PATH + t_cfg.MODEL_NAME)
        BEST_ACCURACY = curr_accuracy
        print("Model is saved !")


########################################################## Graphs ##########################################################
if t_cfg.PLOT_GRAPH:
    plot_graph(t_cfg.EPOCH, "Epoch", "Training Loss", "Training Loss for %d epoch"%(t_cfg.EPOCH), "./loss.png", [ENTIRE_LOSS_LIST, 'r--', "Loss"])
    plot_graph(t_cfg.EPOCH, "Epoch", "Training Accuracy", "Training Accuracy for %d epoch"%(t_cfg.EPOCH), "./accuracy.png", [ENTIRE_ACCURACY_LIST, 'b--', "Accuracy"])
