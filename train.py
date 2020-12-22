'''
Model training script.
'''

import sys
import os
from tqdm import tqdm
import cv2
import numpy as np
import torch
from model import DeepCapsModel
from load_data import FashionMNIST, Cifar10
from helpers import onehot_encode, accuracy_calc
import cfg



train_loader, test_loader, img_size = Cifar10(train_path=cfg.TRAIN_DATASET_PATH,
                                              test_path=cfg.TEST_DATASET_PATH,
                                              batch_size=cfg.BATCH_SIZE,
                                              shuffle=False)()


def train(img_size, device=torch.device('cpu'), learning_rate=1e-3, batch_size=32, num_epochs=100, decay_step=10, gamma=0.98,
          num_classes=10, checkpoint_path=None):
    '''
    Function to train the DeepCaps Model
    '''
    deepcaps = DeepCapsModel(num_class=num_classes, img_height=img_size, img_width=img_size, device=device).to(device) #initialize model

    if not checkpoint_path is None and os.path.exists(checkpoint_path):
        try:
            deepcaps.load_state_dict(torch.load(checkpoint_path))
            print("Checkpoint loaded!")
        except Exception as e:
            print(e)
            sys.exit()

    optimizer = torch.optim.Adam(deepcaps.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=decay_step, gamma=gamma)

    for epoch_idx in range(num_epochs):

        batch_loss = 0
        batch_accuracy = 0
        batch_idx = 0

        for batch_idx, (train_data, labels) in tqdm(enumerate(train_loader)):

            data, labels = train_data.to(device), labels.to(device)

            onehot_label = onehot_encode(labels, num_classes=num_classes, device=device)

            optimizer.zero_grad()

            outputs, masked, reconstructed, indices = deepcaps(data, onehot_label)

            loss = deepcaps.loss(x=outputs, reconstructed=reconstructed, data=data, labels=onehot_label)

            loss.backward()

            optimizer.step()

            batch_loss += loss.item()
            batch_accuracy += accuracy_calc(predictions=indices, labels=labels)


        print(f"Epoch : {epoch_idx}, Accuracy : {batch_accuracy/(batch_idx+1)}, Total Loss : {batch_loss}")
            # resize_img = cv2.resize(data[0].cpu().permute(1,2,0).numpy(), (64,64))
            # cv2.imshow("Ori image", resize_img)
            # cv2.waitKey(0)

            # print(reconstructed.size())
            # cv2.imshow("recon", reconstructed[0].permute(1,2,0).detach().cpu().numpy())
            # cv2.waitKey(0)

            # break



        # cv2.destroyAllWindows()













train(img_size=img_size, device=cfg.DEVICE)










