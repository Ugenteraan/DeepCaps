'''
Model training script.
'''

import sys
import os
import torch
from model import DeepCapsModel
from load_data import FashionMNIST
from helpers import onehot_encode
import cfg



train_loader, test_loader = FashionMNIST(train_path=cfg.TRAIN_DATASET_PATH, test_path=cfg.TEST_DATASET_PATH, batch_size=cfg.BATCH_SIZE, shuffle=True)()


def train(device=torch.device('cpu'), learning_rate=1e-4, batch_size=32, num_epochs=100, decay_step=10, gamma=0.98, num_classes=10, checkpoint_path=None):
    '''
    Function to train the DeepCaps Model
    '''
    deepcaps = DeepCapsModel(num_class=num_classes).to(device) #initialize model

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

        for batch_idx, (train_data, label) in enumerate(train_loader):

            data, label = train_data.to(device), label.to(device)

            label = onehot_encode(label, num_classes=num_classes, device=device)

            optimizer.zero_grad()

            outputs, masked, reconstructed, indices = deepcaps(data, label)

            cv2.imshow("ori img", data[0].cpu().permute(1,2,0).numpy())
            cv2.waitKey(0)

            cv2.imshow("reconstructed", reconstructed[0].cpu().permute(1,2,0).numpy())
            cv2.waitKey(0)













train()










