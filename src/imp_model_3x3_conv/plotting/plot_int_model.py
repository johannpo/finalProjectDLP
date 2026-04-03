import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("/scratch/s4099265/dlp_project/final_project/output/final_data")
OUT_DIR = Path("/scratch/s4099265/dlp_project/final_project/output")

BASE_TRAIN = OUTPUT_DIR / "baseline_fno_paper_N1000_ep500_m12_w20_S64_train.txt"
BASE_TEST  = OUTPUT_DIR / "baseline_fno_paper_N1000_ep500_m12_w20_S64_test.txt"
BASE_LABEL = "Baseline"

# The output from the improved script
IMP_TRAIN  = OUTPUT_DIR / "improved_fno_paper_N1000_ep500_m12_w20_S64_train.txt"
IMP_TEST   = OUTPUT_DIR / "improved_fno_paper_N1000_ep500_m12_w20_S64_test.txt"
IMP_LABEL  = "Improved"

SAVE_PLOT = OUTPUT_DIR / "error_comparison_models.png"

def check_exists(path):
    if not path.exists():
        print(f"Warning: {path} not found. Skipping this file.")
        return False
    return True

def load_metric_file(path):
    if not check_exists(path):
        return None
    arr = np.loadtxt(path, skiprows=1)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr

def extract_metrics(arr):
    if arr is None: return None
    return {
        "epoch": arr[:, 0],
        "step": arr[:, 1],
        "full": arr[:, 2],
    }

def summarize(label, train_metrics, test_metrics):
    if test_metrics is None or train_metrics is None:
        return
    best_idx = np.argmin(test_metrics["full"])
    print(f"\n{label}")
    print(f"  best test full : {test_metrics['full'][best_idx]:.6f} at epoch {int(test_metrics['epoch'][best_idx])}")
    print(f"  final test full: {test_metrics['full'][-1]:.6f}")
    print(f"  final test step: {test_metrics['step'][-1]:.6f}")
    print(f"  final train full: {train_metrics['full'][-1]:.6f}")
    print(f"  final train step: {train_metrics['step'][-1]:.6f}")

# Load all data
data = {
    BASE_LABEL: {
        "train": extract_metrics(load_metric_file(BASE_TRAIN)),
        "test": extract_metrics(load_metric_file(BASE_TEST))
    },
    IMP_LABEL: {
        "train": extract_metrics(load_metric_file(IMP_TRAIN)),
        "test": extract_metrics(load_metric_file(IMP_TRAIN))
    }
}

# Summarize in console
for label, m in data.items():
    summarize(label, m["train"], m["test"])

# Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Subplot 1: Train Step
ax = axes[0, 0]
for label, m in data.items():
    if m["train"] is not None:
        ax.plot(m["train"]["epoch"], m["train"]["step"], label=label, linewidth=1.5)
ax.set_title("Train step error")
ax.set_xlabel("Epoch")
ax.set_ylabel("Relative L2")
ax.set_yscale("log")
ax.grid(True, alpha=0.3)
ax.legend()

# Subplot 2: Train Full
ax = axes[0, 1]
for label, m in data.items():
    if m["train"] is not None:
        ax.plot(m["train"]["epoch"], m["train"]["full"], label=label, linewidth=1.5)
ax.set_title("Train full rollout error")
ax.set_xlabel("Epoch")
ax.set_ylabel("Relative L2")
ax.set_yscale("log")
ax.grid(True, alpha=0.3)
ax.legend()

# Subplot 3: Test Step
ax = axes[1, 0]
for label, m in data.items():
    if m["test"] is not None:
        ax.plot(m["test"]["epoch"], m["test"]["step"], label=label, linewidth=1.5)
ax.set_title("Test step error")
ax.set_xlabel("Epoch")
ax.set_ylabel("Relative L2")
ax.set_yscale("log")
ax.grid(True, alpha=0.3)
ax.legend()

# Subplot 4: Test Full
ax = axes[1, 1]
for label, m in data.items():
    if m["test"] is not None:
        ax.plot(m["test"]["epoch"], m["test"]["full"], label=label, linewidth=1.5)
ax.set_title("Test full rollout error")
ax.set_xlabel("Epoch")
ax.set_ylabel("Relative L2")
ax.set_yscale("log")
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig(SAVE_PLOT, dpi=150)
plt.close()

print(f"\nSaved plot to: {SAVE_PLOT}")
