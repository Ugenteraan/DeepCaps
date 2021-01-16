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
from helpers import onehot_encode, accuracy_calc, get_learning_rate
from plot import plot_loss_acc, plot_reconstruction
import cfg



train_loader, test_loader, img_size, num_class = Cifar10(data_path=cfg.DATASET_FOLDER,
                                                              batch_size=cfg.BATCH_SIZE,
                                                              shuffle=True)()


def train(img_size, device=torch.device('cpu'), learning_rate=1e-3, num_epochs=500, decay_step=5, gamma=0.98,
          num_classes=10, lambda_=0.5, m_plus=0.9, m_minus=0.1, checkpoint_folder=None, checkpoint_name=None, load_checkpoint=False, graphs_folder=None):
    '''
    Function to train the DeepCaps Model
    '''
    checkpoint_path = checkpoint_folder + checkpoint_name

    deepcaps = DeepCapsModel(num_class=num_classes, img_height=img_size, img_width=img_size, device=device).to(device) #initialize model

    #load the current checkpoint
    if load_checkpoint and not checkpoint_name is None and os.path.exists(checkpoint_path):
        try:
            deepcaps.load_state_dict(torch.load(checkpoint_path))
            print("Checkpoint loaded!")
        except Exception as e:
            print(e)
            sys.exit()

    optimizer = torch.optim.Adam(deepcaps.parameters(), lr=learning_rate)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=decay_step, gamma=gamma)

    best_accuracy = 0

    training_loss_list = []
    training_acc_list = []
    testing_loss_list = []
    testing_acc_list = []

    #training and testing
    for epoch_idx in range(num_epochs):


        print(f"Training and testing for epoch {epoch_idx} began with LR : {get_learning_rate(optimizer)}")
        #Training
        batch_loss = 0
        batch_accuracy = 0
        batch_idx = 0

        deepcaps.train() #train mode
        for batch_idx, (train_data, labels) in tqdm(enumerate(train_loader)): #from training dataset

            data, labels = train_data.to(device), labels.to(device)
            onehot_label = onehot_encode(labels, num_classes=num_classes, device=device) #convert the labels into one-hot vectors.

            optimizer.zero_grad()

            outputs, _, reconstructed, indices = deepcaps(data, onehot_label)
            loss = deepcaps.loss(x=outputs, reconstructed=reconstructed, data=data, labels=onehot_label, lambda_=lambda_, m_plus=m_plus, m_minus=m_minus)

            loss.backward()
            optimizer.step()

            batch_loss += loss.item()
            batch_accuracy += accuracy_calc(predictions=indices, labels=labels)


        epoch_accuracy = batch_accuracy/(batch_idx+1)
        avg_batch_loss = batch_loss/(batch_idx+1)
        print(f"Epoch : {epoch_idx}, Training Accuracy : {epoch_accuracy}, Training Loss : {avg_batch_loss}")

        training_loss_list.append(avg_batch_loss)
        training_acc_list.append(epoch_accuracy)


        #Testing
        batch_loss = 0
        batch_accuracy = 0
        batch_idx = 0

        deepcaps.eval() #eval mode
        for batch_idx, (test_data, labels) in tqdm(enumerate(test_loader)): #from testing dataset


            data, labels = test_data.to(device), labels.to(device)
            onehot_label = onehot_encode(labels, num_classes=num_classes, device=device)

            outputs, _, reconstructed, indices = deepcaps(data, onehot_label)
            loss = deepcaps.loss(x=outputs, reconstructed=reconstructed, data=data, labels=onehot_label,  lambda_=lambda_, m_plus=m_plus, m_minus=m_minus)

            batch_loss += loss.item()
            batch_accuracy += accuracy_calc(predictions=indices, labels=labels)



        epoch_accuracy = batch_accuracy/(batch_idx+1)
        avg_batch_loss = batch_loss/(batch_idx+1)
        print(f"Epoch : {epoch_idx}, Testing Accuracy : {epoch_accuracy}, Testing Loss : {avg_batch_loss}")

        testing_loss_list.append(avg_batch_loss)
        testing_acc_list.append(epoch_accuracy)

        # lr_scheduler.step()

        if not graphs_folder is None and epoch_idx%5==0:
            plot_loss_acc(path=graphs_folder, num_epoch=epoch_idx, train_accuracies=training_acc_list, train_losses=training_loss_list,
                          test_accuracies=testing_acc_list, test_losses=testing_loss_list)

            plot_reconstruction(path=graphs_folder, num_epoch=epoch_idx, original_images=data.detach(), reconstructed_images=reconstructed.detach(),
                                predicted_classes=indices.detach(), true_classes=labels.detach())



        if best_accuracy < epoch_accuracy:

            torch.save(deepcaps.state_dict(), checkpoint_path)
            print("Saved model at epoch %d"%(epoch_idx))




if __name__ == '__main__':

    train(img_size=img_size, device=cfg.DEVICE, learning_rate=cfg.LEARNING_RATE, num_epochs=cfg.NUM_EPOCHS, decay_step=cfg.DECAY_STEP, gamma=cfg.DECAY_GAMMA,
          num_classes=num_class, lambda_=cfg.LAMBDA_, m_plus=cfg.M_PLUS, m_minus=cfg.M_MINUS, checkpoint_folder=cfg.CHECKPOINT_FOLDER,
          checkpoint_name=cfg.CHECKPOINT_NAME, load_checkpoint=False, graphs_folder=cfg.GRAPHS_FOLDER)

