'''
Configurations.
'''


import helpers

NUM_CLASS = 10
BATCH_SIZE = 32
CHECKPOINT_PATH = './deepcaps.pth'
TRAIN_DATASET_PATH = './dataset_folder/train/'
TEST_DATASET_PATH = './dataset_folder/test/'
DEVICE = helpers.get_device()

helpers.check_path(TRAIN_DATASET_PATH)
helpers.check_path(TEST_DATASET_PATH)

