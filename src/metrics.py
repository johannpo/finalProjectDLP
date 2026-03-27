import numpy as np
from pathlib import Path


# ============================================================
# 1. USER CONFIG
# ============================================================

RUN_FILES = {
    'baseline': Path('/scratch/s4099265/dlp_project/final_project/output/final_data/eval/baseline_fno_paper_N1000_ep500_m12_w20_S64_step10_predictions.npz'),
    'improved': Path('/scratch/s4099265/dlp_project/final_project/output/final_data/eval/improved_fno_N1000_ep500_m12_w20_S64_step10_predictions.npz'),
}

OUTPUT_DIR = Path('/scratch/s4099265/dlp_project/final_project/output/final_data/metrics')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EPS = 1e-12

# High-frequency mask cutoff as a fraction of max radial frequency.
# Example: 0.5 means "upper half" of the radial spectrum.
HIGH_FREQ_CUTOFF_RATIO = 0.5


# ============================================================
# 2. HELPERS
# ============================================================

def ensure_4d(arr):
    """
    Ensure shape is [N, X, Y, T].
    """
    if arr.ndim == 3:
        return arr[None, ...]
    if arr.ndim != 4:
        raise ValueError(f'Expected 3D or 4D array, got shape {arr.shape}')
    return arr


def rel_l2(a, b, axis=None, eps=EPS):
    num = np.sqrt(np.sum((a - b) ** 2, axis=axis))
    den = np.sqrt(np.sum(b ** 2, axis=axis)) + eps
    return num / den


def spatial_gradients(u):
    """
    u shape: [N, X, Y, T]
    Returns:
        dx: [N, X-1, Y, T]
        dy: [N, X, Y-1, T]
    """
    dx = np.diff(u, axis=1)
    dy = np.diff(u, axis=2)
    return dx, dy


def build_high_freq_mask(nx, ny, cutoff_ratio):
    """
    Build a 2D high-frequency mask for fft2 output.
    Returned mask shape: [X, Y]
    """
    kx = np.fft.fftfreq(nx)
    ky = np.fft.fftfreq(ny)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='ij')
    kr = np.sqrt(kx_grid ** 2 + ky_grid ** 2)

    kr_max = kr.max()
    cutoff = cutoff_ratio * kr_max
    mask = kr >= cutoff
    return mask


def spectral_high_freq_energy(u, hf_mask):
    """
    u shape: [N, X, Y, T]
    hf_mask shape: [X, Y]

    Returns:
        energy: [N, T]
    """
    # Move time forward for easier looping if needed, but fft2 can be vectorized.
    # FFT over spatial axes only.
    u_fft = np.fft.fft2(u, axes=(1, 2))
    power = np.abs(u_fft) ** 2

    # Broadcast mask to [1, X, Y, 1]
    masked_power = power * hf_mask[None, :, :, None]
    energy = np.sum(masked_power, axis=(1, 2))
    return energy


def compute_metrics(pred, true, hf_cutoff_ratio):
    """
    pred, true shapes: [N, X, Y, T]
    Returns a dict of arrays.
    """
    pred = ensure_4d(pred)
    true = ensure_4d(true)

    if pred.shape != true.shape:
        raise ValueError(f'Shape mismatch: pred {pred.shape}, true {true.shape}')

    n, nx, ny, nt = pred.shape

    # --------------------------------------------------------
    # 1) Full-rollout relative L2
    # --------------------------------------------------------
    full_rel_l2_per_sample = rel_l2(
        pred.reshape(n, -1),
        true.reshape(n, -1),
        axis=1,
    )
    full_rel_l2_mean = np.mean(full_rel_l2_per_sample)
    full_rel_l2_std = np.std(full_rel_l2_per_sample)

    # --------------------------------------------------------
    # 2) Per-timestep relative L2
    # --------------------------------------------------------
    # shape -> [N, T]
    time_rel_l2_per_sample = rel_l2(
        np.transpose(pred, (0, 3, 1, 2)),
        np.transpose(true, (0, 3, 1, 2)),
        axis=(2, 3),
    )
    time_rel_l2_mean = np.mean(time_rel_l2_per_sample, axis=0)
    time_rel_l2_std = np.std(time_rel_l2_per_sample, axis=0)

    # --------------------------------------------------------
    # 3) Gradient relative error
    # --------------------------------------------------------
    pred_dx, pred_dy = spatial_gradients(pred)
    true_dx, true_dy = spatial_gradients(true)

    # Per-sample full gradient error
    grad_num = (
        np.sum((pred_dx - true_dx) ** 2, axis=(1, 2, 3)) +
        np.sum((pred_dy - true_dy) ** 2, axis=(1, 2, 3))
    )
    grad_den = (
        np.sum(true_dx ** 2, axis=(1, 2, 3)) +
        np.sum(true_dy ** 2, axis=(1, 2, 3))
    )
    grad_rel_err_per_sample = np.sqrt(grad_num) / (np.sqrt(grad_den) + EPS)
    grad_rel_err_mean = np.mean(grad_rel_err_per_sample)
    grad_rel_err_std = np.std(grad_rel_err_per_sample)

    # Per-timestep gradient error: [N, T]
    grad_num_t = (
        np.sum((pred_dx - true_dx) ** 2, axis=(1, 2)) +
        np.sum((pred_dy - true_dy) ** 2, axis=(1, 2))
    )
    grad_den_t = (
        np.sum(true_dx ** 2, axis=(1, 2)) +
        np.sum(true_dy ** 2, axis=(1, 2))
    )
    grad_rel_err_per_time_sample = np.sqrt(grad_num_t) / (np.sqrt(grad_den_t) + EPS)
    grad_rel_err_per_time_mean = np.mean(grad_rel_err_per_time_sample, axis=0)
    grad_rel_err_per_time_std = np.std(grad_rel_err_per_time_sample, axis=0)

    # --------------------------------------------------------
    # 4) High-frequency spectral error
    # --------------------------------------------------------
    hf_mask = build_high_freq_mask(nx, ny, hf_cutoff_ratio)

    pred_hf_energy = spectral_high_freq_energy(pred, hf_mask)   # [N, T]
    true_hf_energy = spectral_high_freq_energy(true, hf_mask)   # [N, T]

    hf_rel_err_per_time_sample = np.abs(pred_hf_energy - true_hf_energy) / (np.abs(true_hf_energy) + EPS)
    hf_rel_err_per_time_mean = np.mean(hf_rel_err_per_time_sample, axis=0)
    hf_rel_err_per_time_std = np.std(hf_rel_err_per_time_sample, axis=0)

    # Full-rollout high-frequency energy error per sample
    pred_hf_energy_total = np.sum(pred_hf_energy, axis=1)
    true_hf_energy_total = np.sum(true_hf_energy, axis=1)

    hf_rel_err_per_sample = np.abs(pred_hf_energy_total - true_hf_energy_total) / (np.abs(true_hf_energy_total) + EPS)
    hf_rel_err_mean = np.mean(hf_rel_err_per_sample)
    hf_rel_err_std = np.std(hf_rel_err_per_sample)

    return {
        # Metadata
        'shape': np.array(pred.shape),
        'high_freq_cutoff_ratio': np.array(hf_cutoff_ratio),

        # 1) Full rollout relative L2
        'full_rel_l2_per_sample': full_rel_l2_per_sample,
        'full_rel_l2_mean': np.array(full_rel_l2_mean),
        'full_rel_l2_std': np.array(full_rel_l2_std),

        # 2) Per-timestep relative L2
        'time_rel_l2_per_sample': time_rel_l2_per_sample,
        'time_rel_l2_mean': time_rel_l2_mean,
        'time_rel_l2_std': time_rel_l2_std,

        # 3) Gradient relative error
        'grad_rel_err_per_sample': grad_rel_err_per_sample,
        'grad_rel_err_mean': np.array(grad_rel_err_mean),
        'grad_rel_err_std': np.array(grad_rel_err_std),
        'grad_rel_err_per_time_sample': grad_rel_err_per_time_sample,
        'grad_rel_err_per_time_mean': grad_rel_err_per_time_mean,
        'grad_rel_err_per_time_std': grad_rel_err_per_time_std,

        # 4) High-frequency spectral error
        'hf_rel_err_per_sample': hf_rel_err_per_sample,
        'hf_rel_err_mean': np.array(hf_rel_err_mean),
        'hf_rel_err_std': np.array(hf_rel_err_std),
        'hf_rel_err_per_time_sample': hf_rel_err_per_time_sample,
        'hf_rel_err_per_time_mean': hf_rel_err_per_time_mean,
        'hf_rel_err_per_time_std': hf_rel_err_per_time_std,

        # Optional saved intermediates for later analysis
        'pred_hf_energy': pred_hf_energy,
        'true_hf_energy': true_hf_energy,
    }


# ============================================================
# 3. MAIN
# ============================================================

if __name__ == '__main__':
    print('=' * 60)
    print('Computing metrics')
    print('=' * 60)

    for run_name, file_path in RUN_FILES.items():
        print(f'\nRun: {run_name}')
        print(f'Input: {file_path}')

        if not file_path.exists():
            raise FileNotFoundError(f'Missing file: {file_path}')

        data = np.load(file_path)
        pred = data['pred_sol']
        true = data['true_sol']

        results = compute_metrics(pred, true, HIGH_FREQ_CUTOFF_RATIO)

        out_file = OUTPUT_DIR / f'{run_name}_metrics.npz'
        np.savez(out_file, **results)

        print(f'Saved metrics to: {out_file}')
        print(f'  full_rel_l2_mean = {results["full_rel_l2_mean"]:.6f}')
        print(f'  grad_rel_err_mean = {results["grad_rel_err_mean"]:.6f}')
        print(f'  hf_rel_err_mean   = {results["hf_rel_err_mean"]:.6f}')

    print('\nDone.')
