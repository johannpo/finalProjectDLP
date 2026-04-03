import numpy as np
from pathlib import Path


# ============================================================
# 1. USER CONFIG
# ============================================================

OUTPUT_DIR = Path('/scratch/s4099265/dlp_project/final_project/output/metrics')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Give this comparison a name
COMPARISON_NAME = 'step1_nu1e-3'

# Baseline run
BASELINE = {
    'label': 'Baseline',
    'file': Path('/scratch/s4099265/dlp_project/final_project/output/baseline_roll/eval/baseline_roll_samp1000_ep500_m12_w20_tin10_tout40_predictions.npz'),
}

# Other runs to compare against the baseline
RUNS = [
    {
        'label': 'Improved A',
        'file': Path('/scratch/s4099265/dlp_project/final_project/output/improved_one_3x3/eval/improved_one_3x3_samp1000_ep500_m12_w20_tin10_tout40_step1_predictions.npz'),
    },
    {
        'label': 'Improved B',
        'file': Path('/scratch/s4099265/dlp_project/final_project/output/imp_model_ffd_one/eval/imp_model_ffd_one_samp1000_ep500_m12_w20_tin10_tout40_step1_predictions.npz'),
    },
]

EPS = 1e-12
HIGH_FREQ_CUTOFF_RATIO = 0.5

# Write a latex-ready table body
WRITE_LATEX_ROWS = True
LATEX_PATH = OUTPUT_DIR / f'{COMPARISON_NAME}_table_rows.txt'


# ============================================================
# 2. HELPERS
# ============================================================

def ensure_4d(arr):
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
    dx = np.diff(u, axis=1)
    dy = np.diff(u, axis=2)
    return dx, dy


def build_high_freq_mask(nx, ny, cutoff_ratio):
    kx = np.fft.fftfreq(nx)
    ky = np.fft.fftfreq(ny)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='ij')
    kr = np.sqrt(kx_grid ** 2 + ky_grid ** 2)

    kr_max = kr.max()
    cutoff = cutoff_ratio * kr_max
    return kr >= cutoff


def spectral_energies(u, hf_mask):
    u_fft = np.fft.fft2(u, axes=(1, 2))
    power = np.abs(u_fft) ** 2

    hf_energy = np.sum(power * hf_mask[None, :, :, None], axis=(1, 2))
    total_energy = np.sum(power, axis=(1, 2))
    return hf_energy, total_energy


def load_prediction_file(path):
    if not path.exists():
        raise FileNotFoundError(f'Missing file: {path}')

    data = np.load(path)
    if 'pred_sol' not in data or 'true_sol' not in data:
        raise KeyError(f'{path} must contain pred_sol and true_sol')

    pred = ensure_4d(data['pred_sol'])
    true = ensure_4d(data['true_sol'])

    if pred.shape != true.shape:
        raise ValueError(f'Shape mismatch inside {path}: pred {pred.shape}, true {true.shape}')

    return pred, true


def compute_metrics(pred, true, hf_cutoff_ratio):
    pred = ensure_4d(pred)
    true = ensure_4d(true)

    if pred.shape != true.shape:
        raise ValueError(f'Shape mismatch: pred {pred.shape}, true {true.shape}')

    n, nx, ny, nt = pred.shape

    # full rollout relative L2
    full_rel_l2_per_sample = rel_l2(
        pred.reshape(n, -1),
        true.reshape(n, -1),
        axis=1,
    )
    full_rel_l2_mean = np.mean(full_rel_l2_per_sample)
    full_rel_l2_std = np.std(full_rel_l2_per_sample)

    # per-time relative L2
    time_rel_l2_per_sample = rel_l2(
        np.transpose(pred, (0, 3, 1, 2)),
        np.transpose(true, (0, 3, 1, 2)),
        axis=(2, 3),
    )
    time_rel_l2_mean = np.mean(time_rel_l2_per_sample, axis=0)
    time_rel_l2_std = np.std(time_rel_l2_per_sample, axis=0)

    # gradient relative error
    pred_dx, pred_dy = spatial_gradients(pred)
    true_dx, true_dy = spatial_gradients(true)

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

    # high-frequency relative error
    hf_mask = build_high_freq_mask(nx, ny, hf_cutoff_ratio)

    pred_hf_energy, pred_total_energy = spectral_energies(pred, hf_mask)
    true_hf_energy, true_total_energy = spectral_energies(true, hf_mask)

    hf_rel_err_per_time_sample = (
        np.abs(pred_hf_energy - true_hf_energy) / (true_total_energy + EPS)
    )
    hf_rel_err_per_time_mean = np.mean(hf_rel_err_per_time_sample, axis=0)
    hf_rel_err_per_time_std = np.std(hf_rel_err_per_time_sample, axis=0)

    pred_hf_energy_total = np.sum(pred_hf_energy, axis=1)
    true_hf_energy_total = np.sum(true_hf_energy, axis=1)
    true_total_energy_total = np.sum(true_total_energy, axis=1)

    hf_rel_err_per_sample = (
        np.abs(pred_hf_energy_total - true_hf_energy_total) / (true_total_energy_total + EPS)
    )
    hf_rel_err_mean = np.mean(hf_rel_err_per_sample)
    hf_rel_err_std = np.std(hf_rel_err_per_sample)

    return {
        'shape': np.array(pred.shape),
        'high_freq_cutoff_ratio': np.array(hf_cutoff_ratio),

        'full_rel_l2_per_sample': full_rel_l2_per_sample,
        'full_rel_l2_mean': np.array(full_rel_l2_mean),
        'full_rel_l2_std': np.array(full_rel_l2_std),

        'time_rel_l2_per_sample': time_rel_l2_per_sample,
        'time_rel_l2_mean': time_rel_l2_mean,
        'time_rel_l2_std': time_rel_l2_std,

        'grad_rel_err_per_sample': grad_rel_err_per_sample,
        'grad_rel_err_mean': np.array(grad_rel_err_mean),
        'grad_rel_err_std': np.array(grad_rel_err_std),
        'grad_rel_err_per_time_sample': grad_rel_err_per_time_sample,
        'grad_rel_err_per_time_mean': grad_rel_err_per_time_mean,
        'grad_rel_err_per_time_std': grad_rel_err_per_time_std,

        'hf_rel_err_per_sample': hf_rel_err_per_sample,
        'hf_rel_err_mean': np.array(hf_rel_err_mean),
        'hf_rel_err_std': np.array(hf_rel_err_std),
        'hf_rel_err_per_time_sample': hf_rel_err_per_time_sample,
        'hf_rel_err_per_time_mean': hf_rel_err_per_time_mean,
        'hf_rel_err_per_time_std': hf_rel_err_per_time_std,

        'pred_hf_energy': pred_hf_energy,
        'true_hf_energy': true_hf_energy,
        'pred_total_energy': pred_total_energy,
        'true_total_energy': true_total_energy,
    }


def improvement_percent(base_val, new_val, eps=EPS):
    return 100.0 * (base_val - new_val) / (base_val + eps)


def format_row(label, metrics, base_metrics=None):
    full_rel = float(metrics['full_rel_l2_mean'])
    grad_rel = float(metrics['grad_rel_err_mean'])
    hf_rel = float(metrics['hf_rel_err_mean'])

    if base_metrics is None:
        full_imp = 0.0
        grad_imp = 0.0
        hf_imp = 0.0
    else:
        full_imp = improvement_percent(float(base_metrics['full_rel_l2_mean']), full_rel)
        grad_imp = improvement_percent(float(base_metrics['grad_rel_err_mean']), grad_rel)
        hf_imp = improvement_percent(float(base_metrics['hf_rel_err_mean']), hf_rel)

    return {
        'label': label,
        'full_rel_l2_mean': full_rel,
        'grad_rel_err_mean': grad_rel,
        'hf_rel_err_mean': hf_rel,
        'full_rel_l2_improvement_pct': full_imp,
        'grad_rel_err_improvement_pct': grad_imp,
        'hf_rel_err_improvement_pct': hf_imp,
    }


# ============================================================
# 3. MAIN
# ============================================================

if __name__ == '__main__':
    print('=' * 60)
    print('Comparing runs against matched baseline')
    print('=' * 60)

    baseline_pred, baseline_true = load_prediction_file(BASELINE['file'])
    baseline_metrics = compute_metrics(baseline_pred, baseline_true, HIGH_FREQ_CUTOFF_RATIO)

    rows = []
    rows.append(format_row(BASELINE['label'], baseline_metrics, base_metrics=None))

    print('\nBaseline:')
    print(f'  Label: {BASELINE["label"]}')
    print(f'  File : {BASELINE["file"]}')
    print(f'  full_rel_l2_mean = {float(baseline_metrics["full_rel_l2_mean"]):.6f}')
    print(f'  grad_rel_err_mean = {float(baseline_metrics["grad_rel_err_mean"]):.6f}')
    print(f'  hf_rel_err_mean   = {float(baseline_metrics["hf_rel_err_mean"]):.6f}')

    for run in RUNS:
        pred, true = load_prediction_file(run['file'])

        if pred.shape != baseline_pred.shape:
            raise ValueError(
                f'Shape mismatch between baseline and {run["label"]}: '
                f'{baseline_pred.shape} vs {pred.shape}'
            )

        metrics = compute_metrics(pred, true, HIGH_FREQ_CUTOFF_RATIO)
        row = format_row(run['label'], metrics, base_metrics=baseline_metrics)
        rows.append(row)

        out_file = OUTPUT_DIR / f'{COMPARISON_NAME}_{run["label"].lower().replace(" ", "_")}_metrics.npz'
        np.savez(out_file, **metrics)

        print(f'\nRun: {run["label"]}')
        print(f'  File                     : {run["file"]}')
        print(f'  Saved metrics            : {out_file}')
        print(f'  full_rel_l2_mean         = {row["full_rel_l2_mean"]:.6f}')
        print(f'  grad_rel_err_mean        = {row["grad_rel_err_mean"]:.6f}')
        print(f'  hf_rel_err_mean          = {row["hf_rel_err_mean"]:.6f}')
        print(f'  full_rel_l2_improv_pct   = {row["full_rel_l2_improvement_pct"]:.2f}')
        print(f'  grad_rel_err_improv_pct  = {row["grad_rel_err_improvement_pct"]:.2f}')
        print(f'  hf_rel_err_improv_pct    = {row["hf_rel_err_improvement_pct"]:.2f}')

    summary_path = OUTPUT_DIR / f'{COMPARISON_NAME}_comparison_summary.txt'
    with open(summary_path, 'w') as f:
        f.write(f'COMPARISON_NAME = {COMPARISON_NAME}\n')
        f.write(f'BASELINE = {BASELINE["label"]}\n')
        f.write(f'BASELINE_FILE = {BASELINE["file"]}\n')
        f.write('\n')
        for row in rows:
            f.write(f'LABEL = {row["label"]}\n')
            f.write(f'full_rel_l2_mean = {row["full_rel_l2_mean"]}\n')
            f.write(f'grad_rel_err_mean = {row["grad_rel_err_mean"]}\n')
            f.write(f'hf_rel_err_mean = {row["hf_rel_err_mean"]}\n')
            f.write(f'full_rel_l2_improvement_pct = {row["full_rel_l2_improvement_pct"]}\n')
            f.write(f'grad_rel_err_improvement_pct = {row["grad_rel_err_improvement_pct"]}\n')
            f.write(f'hf_rel_err_improvement_pct = {row["hf_rel_err_improvement_pct"]}\n')
            f.write('\n')

    print(f'\nSaved summary to: {summary_path}')

    if WRITE_LATEX_ROWS:
        with open(LATEX_PATH, 'w') as f:
            for row in rows:
                f.write(
                    f'{row["label"]} & '
                    f'{row["full_rel_l2_mean"]:.4f} & '
                    f'{row["grad_rel_err_mean"]:.4f} & '
                    f'{row["hf_rel_err_mean"]:.4f} & '
                    f'{row["full_rel_l2_improvement_pct"]:.1f}\\% \\\\\n'
                )

        print(f'Saved LaTeX rows to: {LATEX_PATH}')

    print('\nDone.')
