import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- CONFIG ---
# Make sure these match the variables you used in the training script
PAPER_NTRAIN = 1000
EPOCHS = 500
DESIRED_MODES = 12
WIDTH = 20

OUTPUT_DIR = Path("/scratch/s4099265/dlp_project/final_project/output/final_data")

RUN_NAME = f"baseline_fno_paper_N{PAPER_NTRAIN}_ep{EPOCHS}_m{DESIRED_MODES}_w{WIDTH}_S64"
TRAIN_FILE = OUTPUT_DIR / f"{RUN_NAME}_train.txt"
TEST_FILE  = OUTPUT_DIR / f"{RUN_NAME}_test.txt"
SAVE_PLOT  = OUTPUT_DIR / f"{RUN_NAME}_loss_plot.png"

# --- UTILS ---
def load_metric_file(path):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Check if the training script finished.")
    
    arr = np.loadtxt(path, skiprows=1)
    if arr.ndim == 1: 
        arr = arr[None, :]
    return {
        "epoch": arr[:, 0],
        "step": arr[:, 1],
        "full": arr[:, 2],
    }

def summarize(label, metrics_train, metrics_test):
    best_idx = np.argmin(metrics_test["full"])
    print(f"\n{label}")
    print(f"  Best Test Full:  {metrics_test['full'][best_idx]:.6f} at epoch {int(metrics_test['epoch'][best_idx])}")
    print(f"  Final Test Full: {metrics_test['full'][-1]:.6f}")
    print(f"  Final Test Step: {metrics_test['step'][-1]:.6f}")
    print(f"  Final Train Full:{metrics_train['full'][-1]:.6f}")
    print(f"  Final Train Step:{metrics_train['step'][-1]:.6f}")

# --- LOAD DATA ---
train_data = load_metric_file(TRAIN_FILE)
test_data = load_metric_file(TEST_FILE)

summarize("Baseline 64x64", train_data, test_data)

# --- PLOT ---
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
label = "Baseline (Li et al.)"

# 1. Train Step Error
axes[0, 0].plot(train_data["epoch"], train_data["step"], label=label, color="tab:blue")
axes[0, 0].set_title("Train step error")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Relative L2")
axes[0, 0].set_yscale("log")
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()

# 2. Train Full Rollout Error
axes[0, 1].plot(train_data["epoch"], train_data["full"], label=label, color="tab:blue")
axes[0, 1].set_title("Train full rollout error")
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("Relative L2")
axes[0, 1].set_yscale("log")
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()

# 3. Test Step Error
axes[1, 0].plot(test_data["epoch"], test_data["step"], label=label, color="tab:orange")
axes[1, 0].set_title("Test step error")
axes[1, 0].set_xlabel("Epoch")
axes[1, 0].set_ylabel("Relative L2")
axes[1, 0].set_yscale("log")
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend()

# 4. Test Full Rollout Error
axes[1, 1].plot(test_data["epoch"], test_data["full"], label=label, color="tab:orange")
axes[1, 1].set_title("Test full rollout error")
axes[1, 1].set_xlabel("Epoch")
axes[1, 1].set_ylabel("Relative L2")
axes[1, 1].set_yscale("log")
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend()

plt.tight_layout()
plt.savefig(SAVE_PLOT, dpi=150)
plt.close()

print(f"\nSaved loss plot to: {SAVE_PLOT}")
