import numpy as np
import matplotlib.pyplot as plt
import os
import time
from pathlib import Path

# --- 1. CONFIGURATION ---
# Matching the exact output path from your Habrok log
PROJROOT = Path('/scratch/s4099265/dlp_project/final_project')
RUN_NAME = 'baseline_fno_paper_N1000_ep500_m12_w20_S64_step10'

DATA_FILE = PROJROOT / 'output' / 'final_data' / 'eval' / f'{RUN_NAME}_predictions.npz'
OUTPUT_DIR = PROJROOT / 'output' / 'final_data' 
NUM_SAMPLES_TO_PLOT = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. FILE VERIFICATION CHECKS ---
print("="*50)
print(" FILE VERIFICATION CHECKS")
print("="*50)
print(f"Target Path: {DATA_FILE}")

if not DATA_FILE.exists():
    raise FileNotFoundError(f"\n[ERROR] File does not exist! Check the path above.")

file_stats = DATA_FILE.stat()
size_mb = file_stats.st_size / (1024 * 1024)
mod_time = time.ctime(file_stats.st_mtime)

print(f"[OK] File Found.")
print(f"     Size:          {size_mb:.2f} MB")
print(f"     Last Modified: {mod_time}")
print("-" * 50)

# --- 3. LOAD AND INSPECT DATA ---
data = np.load(DATA_FILE)
trues = data['true_sol']
preds = data['pred_sol']

print(" DATA SHAPES")
print("-" * 50)
print(f"     Ground Truth Shape: {trues.shape}")
print(f"     Predictions Shape:  {preds.shape}")

# Ensure dimensions are (Samples, X, Y, T)
if trues.ndim == 3:
    trues = np.expand_dims(trues, axis=0)
    preds = np.expand_dims(preds, axis=0)

num_samples = min(NUM_SAMPLES_TO_PLOT, trues.shape[0])
total_timesteps = trues.shape[-1]

# Pick 4 specific timesteps to show the evolution (e.g., T=10, 20, 30, 40)
t_steps = sorted(set([
    max(0, int(total_timesteps * 0.25) - 1),
    max(0, int(total_timesteps * 0.50) - 1),
    max(0, int(total_timesteps * 0.75) - 1),
    total_timesteps - 1,
]))

print(f"\n[INFO] Plotting samples 0 to {num_samples-1} at timesteps: {[t+1 for t in t_steps]}")
print("="*50)

# --- 4. PLOTTING LOOP ---
for i in range(num_samples):
    true_fluid = trues[i]
    pred_fluid = preds[i]
    err_fluid = np.abs(pred_fluid - true_fluid)

    fig, axes = plt.subplots(3, len(t_steps), figsize=(4 * len(t_steps), 10))
    cmap = 'jet'

    # Global min/max across all timesteps for consistent color scaling
    vmin = min(true_fluid.min(), pred_fluid.min())
    vmax = max(true_fluid.max(), pred_fluid.max())
    emax = err_fluid.max()

    for col_idx, t in enumerate(t_steps):
        # ROW 1: Ground Truth
        ax = axes[0, col_idx]
        im0 = ax.imshow(true_fluid[:, :, t], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"Ground Truth t={t+1}")
        ax.set_xticks([])
        ax.set_yticks([])

        # ROW 2: Prediction
        ax = axes[1, col_idx]
        im1 = ax.imshow(pred_fluid[:, :, t], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"Baseline FNO t={t+1}")
        ax.set_xticks([])
        ax.set_yticks([])

        # ROW 3: Absolute Error
        ax = axes[2, col_idx]
        im2 = ax.imshow(err_fluid[:, :, t], cmap='magma', vmin=0.0, vmax=emax)
        ax.set_title(f"|Error| t={t+1}")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    save_path = OUTPUT_DIR / f'{RUN_NAME}_sample_{i:03d}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"-> Saved: {save_path.name}")

print("\nDone plotting.")
