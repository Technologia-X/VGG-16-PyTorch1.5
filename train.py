'''
Training script for VGG-16.
'''
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from model import VGG, OPTIMIZER, LR_DECAY, LOSS_CRITERION
from load_data import LoadDataset, TRAINING_DATA
from utils import calculate_accuracy
import cfg
import cv2

print("--- Model Architecture ---")
print(VGG)
#loads the model if a saved model is present.
if cfg.TRAINED_MODEL_PRESENCE:

    MODEL_PARAMS = torch.load(cfg.MODEL_PATH+cfg.MODEL_NAME) #get
    VGG.load_state_dict(MODEL_PARAMS)
    print("Model parameters are loaded from the saved file!")



DATALOADER = DataLoader(TRAINING_DATA, batch_size=cfg.BATCH_SIZE, shuffle=cfg.DATA_SHUFFLE, num_workers=cfg.NUM_WORKERS)

for epoch_idx in range(cfg.EPOCH):

    print("Training for epoch %d has started!"%(epoch_idx+1))
    epoch_training_loss = []

    for i, sample in tqdm(enumerate(DATALOADER)):

        batch_x, batch_y = sample['image'], sample['label']

        print(batch_x.size())
        for j in range(cfg.BATCH_SIZE):
            cv2.imshow('img', batch_x[j].cpu().numpy().transpose((1, 2, 0)))
            cv2.waitKey(0)


    break
        # OPTIMIZER.zero_grad() #clear the gradients in the optimizer between every batch.

        # net_output = VGG(batch_x) #output from the network.

        # print("DATA")
        # print(batch_y)
        # print(net_output)
        # total_loss = LOSS_CRITERION(input=net_output, target=batch_y)

        # epoch_training_loss.append(total_loss.item()) #append the loss of every batch.

        # total_loss.backward() #calculate the gradients.
        # OPTIMIZER.step()

        # batch_acc = calculate_accuracy(network_output=net_output, target=batch_y)

