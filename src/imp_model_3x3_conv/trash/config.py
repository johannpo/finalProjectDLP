import os

PROJROOT = '/scratch/s4099265/dlp_project/final_project'
DATAROOT = os.path.join(PROJROOT, 'data')

TRAIN_PATH = os.path.join(DATAROOT, 'ns_V1e-4_N10000_T30.mat')
TEST_PATH = os.path.join(DATAROOT, 'ns_V1e-4_N10000_T30.mat')

DESIRED_MODES = 12
WIDTH = 20
DESIRED_BATCH_SIZE = 20
EPOCHS = 500

LEARNING_RATE = 0.001
SCHEDULER_STEP = 100
SCHEDULER_GAMMA = 0.5

SUB = 1
DESIRED_T_IN = 10
DESIRED_T = 40
STEP = 10

NTRAIN = 1000
NTEST = 200

OUTPUT_DIR = os.path.join(PROJROOT, 'output', 'improved_model_V1e-4')
MODEL_DIR = os.path.join(OUTPUT_DIR, 'model')
EVAL_DIR = os.path.join(OUTPUT_DIR, 'eval')

RUN_NAME = f'improved_model_N{NTRAIN}_ep{EPOCHS}_dim64'

PATH_MODEL = os.path.join(MODEL_DIR, RUN_NAME + '.pt')
PATH_TRAIN_ERR = os.path.join(OUTPUT_DIR, RUN_NAME + '_train.txt')
PATH_TEST_ERR = os.path.join(OUTPUT_DIR, RUN_NAME + '_test.txt')
PATH_TEST_DATA = os.path.join(EVAL_DIR, RUN_NAME + '_test_data.pt')
PATH_PREDICTIONS = os.path.join(EVAL_DIR, RUN_NAME + '_predictions.npz')
