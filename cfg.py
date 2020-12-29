'''
Configurations.
'''


import helpers

LEARNING_RATE = 5e-3
NUM_EPOCHS = 501
BATCH_SIZE = 128
LAMBDA_ = 0.5
M_PLUS = 0.9
M_MINUS = 0.1
DECAY_STEP = 20
DECAY_GAMMA = 0.95
CHECKPOINT_FOLDER = './saved_model/'
CHECKPOINT_NAME = 'deepcaps.pth'
DATASET_FOLDER = './dataset_folder/'
GRAPHS_FOLDER = './graphs/'
DEVICE = helpers.get_device()

helpers.check_path(CHECKPOINT_FOLDER)
helpers.check_path(DATASET_FOLDER)
helpers.check_path(GRAPHS_FOLDER)

