import sys
from pathlib import Path
from timeit import default_timer

import numpy as np
import torch

# ============================================================
# 1. PATHS
# ============================================================

# Folder containing config.py and model_components.py
SRC_DIR = Path("/scratch/s4099265/dlp_project/final_project/src/imp_model_ffd")

# Exact checkpoint you want to evaluate
MODEL_PATH = Path(
    "/scratch/s4099265/dlp_project/final_project/output/imp_model_ffd_one/model/imp_model_ffd_one_samp1000_ep500_m12_w20_tin10_tout40_step1.pt"
)

# Add source folder so we can import your existing scripts
sys.path.insert(0, str(SRC_DIR))

from model_components import ImprovedFNODiffLocal2d  # noqa: E402

# ============================================================
# 2. INFER MATCHING PATHS
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Missing checkpoint: {MODEL_PATH}")

MODEL_DIR = MODEL_PATH.parent
RUN_DIR = MODEL_DIR.parent
EVAL_DIR = RUN_DIR / "eval"

RUN_NAME = MODEL_PATH.stem
PATH_TEST_DATA = EVAL_DIR / f"{RUN_NAME}_test_data.pt"
PATH_PREDICTIONS = EVAL_DIR / f"{RUN_NAME}_predictions.npz"

if not PATH_TEST_DATA.exists():
    raise FileNotFoundError(
        f"Missing matching test data pack:\n{PATH_TEST_DATA}\n"
        "Run the matching training script first."
    )

print("MODEL_PATH      =", MODEL_PATH)
print("PATH_TEST_DATA  =", PATH_TEST_DATA)
print("PATH_PREDICTIONS=", PATH_PREDICTIONS)


# ============================================================
# 3. EXECUTION
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    print(f"Loading test data from {PATH_TEST_DATA} ...")
    data_pack = torch.load(PATH_TEST_DATA, map_location="cpu")

    test_a = data_pack["test_a"]
    test_u = data_pack["test_u"]
    T_in = int(data_pack["T_in"])
    T = int(data_pack["T"])
    STEP = int(data_pack["STEP"])
    ntest = int(data_pack["ntest"])

    # infer spatial size and modes from saved data
    S = test_a.shape[1]
    DESIRED_MODES = 12
    WIDTH = 20
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

    print(f"Loading trained weights from {MODEL_PATH} ...")
    model = ImprovedFNODiffLocal2d(modes, WIDTH, T_in, STEP).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    print(f"Generating sequence predictions for {ntest} samples over {T} steps ...")
    t0 = default_timer()
    pred_all = []

    with torch.no_grad():
        for i in range(ntest):
            xx = test_a[i:i + 1].to(device)
            pred = None

            for t in range(0, T, STEP):
                im = model(xx)

                if pred is None:
                    pred = im
                else:
                    pred = torch.cat((pred, im), dim=-1)

                # assumes channel order [u_0, ..., u_9, x, y]
                xx = torch.cat((xx[..., STEP:-2], im, xx[..., -2:]), dim=-1)

            pred_all.append(pred.cpu())

            if (i + 1) % 50 == 0 or (i + 1) == ntest:
                print(f"  Processed {i + 1}/{ntest} samples")

    t1 = default_timer()
    print(f"Evaluation complete in {t1 - t0:.2f} seconds")

    pred_sol = torch.cat(pred_all, dim=0).numpy()
    true_sol = test_u.numpy()

    np.savez(PATH_PREDICTIONS, pred_sol=pred_sol, true_sol=true_sol)
    print(f"Saved prediction arrays to {PATH_PREDICTIONS}")
