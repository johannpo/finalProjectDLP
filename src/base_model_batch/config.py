import os

PROJROOT = "/scratch/s4099265/dlp_project/final_project"

DATAROOT = os.path.join(PROJROOT, "data")
TRAIN_PATH = os.path.join(DATAROOT, "ns_V1e-4_N10000_T30.mat")
TEST_PATH = os.path.join(DATAROOT, "ns_V1e-4_N10000_T30.mat")

DESIRED_MODES = 12
WIDTH = 20
DESIRED_BATCH_SIZE = 20
EPOCHS = 500

LEARNING_RATE = 0.001
SCHEDULER_STEP = 100
SCHEDULER_GAMMA = 0.5

SUB = 1
DESIRED_T_IN = 10
DESIRED_T = 30
STEP = 10

NTRAIN = 1000
NTEST = 200

MODEL_NAME = "baseline_batch_V1e-4"


def get_output_dir(out_dir=None):
    if out_dir is not None:
        return out_dir
    return os.path.join(PROJROOT, "output", MODEL_NAME)


def get_model_dir(out_dir=None):
    return os.path.join(get_output_dir(out_dir), "model")


def get_eval_dir(out_dir=None):
    return os.path.join(get_output_dir(out_dir), "eval")


def get_run_name():
    return (
        f"{MODEL_NAME}"
        f"_samp{NTRAIN}"
        f"_ep{EPOCHS}"
        f"_m{DESIRED_MODES}"
        f"_w{WIDTH}"
        f"_tin{DESIRED_T_IN}"
        f"_tout{DESIRED_T}"
        f"_step{STEP}"
    )


def make_output_dirs(out_dir=None):
    os.makedirs(get_output_dir(out_dir), exist_ok=True)
    os.makedirs(get_model_dir(out_dir), exist_ok=True)
    os.makedirs(get_eval_dir(out_dir), exist_ok=True)


def get_path_model(out_dir=None):
    return os.path.join(get_model_dir(out_dir), get_run_name() + ".pt")


def get_path_train_err(out_dir=None):
    return os.path.join(get_output_dir(out_dir), get_run_name() + "_train.txt")


def get_path_test_err(out_dir=None):
    return os.path.join(get_output_dir(out_dir), get_run_name() + "_test.txt")


def get_path_test_data(out_dir=None):
    return os.path.join(get_eval_dir(out_dir), get_run_name() + "_test_data.pt")


def get_path_predictions(out_dir=None):
    return os.path.join(get_eval_dir(out_dir), get_run_name() + "_predictions.npz")
