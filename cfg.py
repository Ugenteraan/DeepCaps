'''
Configurations.
'''


import helpers

NUM_CLASS = 10
BATCH_SIZE = 64
CHECKPOINT_PATH = './deepcaps.pth'
DATASET_FOLDER = './dataset_folder/'
DEVICE = helpers.get_device()

helpers.check_path(DATASET_FOLDER)

