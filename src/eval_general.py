import importlib.util
import re
from pathlib import Path
from timeit import default_timer

import numpy as np
import torch


# ============================================================
# 1. USER CONFIG
# ============================================================

# Folder containing model_components.py and usually config.py
SRC_DIR = Path("/scratch/s4099265/dlp_project/final_project/src/base_model_batch")

# Exact checkpoint to evaluate
MODEL_PATH = Path(
    "/scratch/s4099265/dlp_project/final_project/output/baseline_batch_V1e-4/model/baseline_batch_V1e-4_samp1000_ep500_m12_w20_tin10_tout30_step10.pt"
)

# Class name inside model_components.py
MODEL_CLASS_NAME = "BaselineFNO2dMultiStep"

# Optional overrides. If None, try to read from config.py in SRC_DIR.
DESIRED_MODES_OVERRIDE = None
WIDTH_OVERRIDE = None
EPOCHS_OVERRIDE = None

TRAIN_STDOUT_LOG_PATH = Path("/scratch/s4099265/dlp_project/final_project/logs/baseline_batch.sh_28282038.out")
# TRAIN_STDOUT_LOG_PATH = Path("/scratch/.../logs/train_imp_model_ffd_one.log")

# Optional manual average epoch time if you do not have a stdout log
MANUAL_TIME_PER_EPOCH = None

WRITE_SUMMARY = True


# ============================================================
# 2. HELPERS
# ============================================================

def load_module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def parse_final_metrics_txt(path):
    """
    Expected format:
    epoch value1 value2
    with header on first line.
    Returns last epoch and final values.
    """
    if not path.exists():
        return None

    lines = path.read_text().strip().splitlines()
    if len(lines) < 2:
        return None

    last = lines[-1].strip().split()
    if len(last) < 3:
        return None

    return {
        "epoch": int(float(last[0])),
        "metric_1": float(last[1]),
        "metric_2": float(last[2]),
    }


def parse_average_epoch_time_from_log(path):
    """
    Parses lines like:
    Epoch: 12 | Time: 4.55s | Train Step: ...
    Returns average epoch time if found.
    """
    if path is None or not path.exists():
        return None

    text = path.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r"Time:\s*([0-9]*\.?[0-9]+)s", text)

    if not matches:
        return None

    vals = [float(x) for x in matches]
    return float(np.mean(vals))


def format_float_or_na(x, fmt=".6f"):
    if x is None:
        return "N/A"
    return format(x, fmt)


# ============================================================
# 3. PATH SETUP
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Missing checkpoint:\n{MODEL_PATH}")

MODEL_DIR = MODEL_PATH.parent
RUN_DIR = MODEL_DIR.parent
EVAL_DIR = RUN_DIR / "eval"
RUN_NAME = MODEL_PATH.stem

PATH_TEST_DATA = EVAL_DIR / f"{RUN_NAME}_test_data.pt"
PATH_PREDICTIONS = EVAL_DIR / f"{RUN_NAME}_predictions.npz"
PATH_SUMMARY = EVAL_DIR / f"{RUN_NAME}_eval_summary.txt"

PATH_TRAIN_TXT = RUN_DIR / f"{RUN_NAME}_train.txt"
PATH_TEST_TXT = RUN_DIR / f"{RUN_NAME}_test.txt"

if not PATH_TEST_DATA.exists():
    raise FileNotFoundError(
        f"Missing matching test data pack:\n{PATH_TEST_DATA}\n"
        "Run the matching training script first."
    )

MODEL_COMPONENTS_PATH = SRC_DIR / "model_components.py"
CONFIG_PATH = SRC_DIR / "config.py"

if not MODEL_COMPONENTS_PATH.exists():
    raise FileNotFoundError(f"Missing model_components.py:\n{MODEL_COMPONENTS_PATH}")

print("SRC_DIR         =", SRC_DIR)
print("MODEL_PATH      =", MODEL_PATH)
print("RUN_DIR         =", RUN_DIR)
print("EVAL_DIR        =", EVAL_DIR)
print("PATH_TEST_DATA  =", PATH_TEST_DATA)
print("PATH_PREDICTIONS=", PATH_PREDICTIONS)
print("PATH_TRAIN_TXT  =", PATH_TRAIN_TXT)
print("PATH_TEST_TXT   =", PATH_TEST_TXT)


# ============================================================
# 4. LOAD SOURCE MODULES
# ============================================================

model_components = load_module_from_file("dynamic_model_components", MODEL_COMPONENTS_PATH)

if not hasattr(model_components, MODEL_CLASS_NAME):
    raise AttributeError(
        f"{MODEL_CLASS_NAME} not found in {MODEL_COMPONENTS_PATH}"
    )

ModelClass = getattr(model_components, MODEL_CLASS_NAME)

config = None
if CONFIG_PATH.exists():
    config = load_module_from_file("dynamic_config", CONFIG_PATH)


# ============================================================
# 5. RESOLVE MODEL HYPERPARAMETERS
# ============================================================

if WIDTH_OVERRIDE is not None:
    WIDTH = WIDTH_OVERRIDE
elif config is not None and hasattr(config, "WIDTH"):
    WIDTH = int(config.WIDTH)
else:
    raise ValueError("WIDTH not provided and not found in config.py")

if DESIRED_MODES_OVERRIDE is not None:
    DESIRED_MODES = DESIRED_MODES_OVERRIDE
elif config is not None and hasattr(config, "DESIRED_MODES"):
    DESIRED_MODES = int(config.DESIRED_MODES)
else:
    raise ValueError("DESIRED_MODES not provided and not found in config.py")

if EPOCHS_OVERRIDE is not None:
    EPOCHS = int(EPOCHS_OVERRIDE)
elif config is not None and hasattr(config, "EPOCHS"):
    EPOCHS = int(config.EPOCHS)
else:
    EPOCHS = None


# ============================================================
# 6. MAIN
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    print(f"Loading test data from {PATH_TEST_DATA} ...")
    data_pack = safe_torch_load(PATH_TEST_DATA, map_location="cpu")

    test_a = data_pack["test_a"]
    test_u = data_pack["test_u"]
    T_in = int(data_pack["T_in"])
    T = int(data_pack["T"])
    STEP = int(data_pack["STEP"])
    ntest = int(data_pack["ntest"])

    S = int(test_a.shape[1])
    modes = min(DESIRED_MODES, max(1, S // 2))

    print("Loaded test pack:")
    print("  test_a.shape =", tuple(test_a.shape))
    print("  test_u.shape =", tuple(test_u.shape))
    print("  T_in         =", T_in)
    print("  T            =", T)
    print("  STEP         =", STEP)
    print("  ntest        =", ntest)
    print("  S            =", S)
    print("  modes        =", modes)
    print("  width        =", WIDTH)

    print(f"Loading trained weights from {MODEL_PATH} ...")
    model = ModelClass(modes, WIDTH, T_in, STEP).to(device)

    state_dict = safe_torch_load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    num_params = int(model.count_params()) if hasattr(model, "count_params") else sum(
        p.numel() for p in model.parameters()
    )
    print("  num_params   =", num_params)

    print(f"Generating sequence predictions for {ntest} samples over {T} steps ...")
    t0 = default_timer()
    pred_all = []

    with torch.no_grad():
        for i in range(ntest):
            xx = test_a[i:i + 1].to(device)
            pred = None

            for _ in range(0, T, STEP):
                im = model(xx)

                if pred is None:
                    pred = im
                else:
                    pred = torch.cat((pred, im), dim=-1)

                # assumes channel order [u_0, ..., u_(T_in-1), x, y]
                xx = torch.cat((xx[..., STEP:-2], im, xx[..., -2:]), dim=-1)

            pred_all.append(pred.cpu())

            if (i + 1) % 50 == 0 or (i + 1) == ntest:
                print(f"  Processed {i + 1}/{ntest} samples")

    t1 = default_timer()
    eval_seconds = t1 - t0
    print(f"Evaluation complete in {eval_seconds:.2f} seconds")

    pred_sol = torch.cat(pred_all, dim=0).numpy()
    true_sol = test_u.numpy()

    np.savez(PATH_PREDICTIONS, pred_sol=pred_sol, true_sol=true_sol)
    print(f"Saved prediction arrays to {PATH_PREDICTIONS}")

    # --------------------------------------------------------
    # Table-ready metadata
    # --------------------------------------------------------
    train_metrics = parse_final_metrics_txt(PATH_TRAIN_TXT)
    test_metrics = parse_final_metrics_txt(PATH_TEST_TXT)

    avg_time_per_epoch = parse_average_epoch_time_from_log(TRAIN_STDOUT_LOG_PATH)
    if avg_time_per_epoch is None and MANUAL_TIME_PER_EPOCH is not None:
        avg_time_per_epoch = float(MANUAL_TIME_PER_EPOCH)

    final_train_step = None
    final_train_full = None
    final_test_step = None
    final_test_full = None
    trained_epochs = None

    if train_metrics is not None:
        trained_epochs = train_metrics["epoch"] + 1
        final_train_step = train_metrics["metric_1"]
        final_train_full = train_metrics["metric_2"]

    if test_metrics is not None:
        final_test_step = test_metrics["metric_1"]
        final_test_full = test_metrics["metric_2"]
        if trained_epochs is None:
            trained_epochs = test_metrics["epoch"] + 1

    print("\n============================================================")
    print("TABLE-READY SUMMARY")
    print("============================================================")
    print("Run name             :", RUN_NAME)
    print("Checkpoint           :", MODEL_PATH)
    print("Model class          :", MODEL_CLASS_NAME)
    print("Parameters           :", num_params)
    print("Viscosity hint       :", "read from path/config manually if needed")
    print("Grid size            :", S)
    print("T_in                 :", T_in)
    print("T                    :", T)
    print("STEP                 :", STEP)
    print("Epochs               :", trained_epochs if trained_epochs is not None else "N/A")
    print("Avg time / epoch     :", format_float_or_na(avg_time_per_epoch, ".2f"), "s")
    print("Final train step err :", format_float_or_na(final_train_step))
    print("Final train full err :", format_float_or_na(final_train_full))
    print("Final test step err  :", format_float_or_na(final_test_step))
    print("Final test full err  :", format_float_or_na(final_test_full))
    print("Eval time total      :", f"{eval_seconds:.2f} s")
    print("Predictions file     :", PATH_PREDICTIONS)

    if WRITE_SUMMARY:
        with open(PATH_SUMMARY, "w") as f:
            f.write(f"RUN_NAME = {RUN_NAME}\n")
            f.write(f"MODEL_PATH = {MODEL_PATH}\n")
            f.write(f"MODEL_CLASS_NAME = {MODEL_CLASS_NAME}\n")
            f.write(f"SRC_DIR = {SRC_DIR}\n")
            f.write(f"PATH_TEST_DATA = {PATH_TEST_DATA}\n")
            f.write(f"PATH_PREDICTIONS = {PATH_PREDICTIONS}\n")
            f.write(f"PATH_TRAIN_TXT = {PATH_TRAIN_TXT}\n")
            f.write(f"PATH_TEST_TXT = {PATH_TEST_TXT}\n")
            f.write(f"TRAIN_STDOUT_LOG_PATH = {TRAIN_STDOUT_LOG_PATH}\n")
            f.write(f"WIDTH = {WIDTH}\n")
            f.write(f"DESIRED_MODES = {DESIRED_MODES}\n")
            f.write(f"modes_used = {modes}\n")
            f.write(f"num_params = {num_params}\n")
            f.write(f"S = {S}\n")
            f.write(f"T_in = {T_in}\n")
            f.write(f"T = {T}\n")
            f.write(f"STEP = {STEP}\n")
            f.write(f"ntest = {ntest}\n")
            f.write(f"epochs = {trained_epochs}\n")
            f.write(f"avg_time_per_epoch_seconds = {avg_time_per_epoch}\n")
            f.write(f"final_train_step_error = {final_train_step}\n")
            f.write(f"final_train_full_error = {final_train_full}\n")
            f.write(f"final_test_step_error = {final_test_step}\n")
            f.write(f"final_test_full_error = {final_test_full}\n")
            f.write(f"eval_seconds = {eval_seconds}\n")
            f.write(f"test_a.shape = {tuple(test_a.shape)}\n")
            f.write(f"test_u.shape = {tuple(test_u.shape)}\n")

        print(f"Saved summary to {PATH_SUMMARY}")
