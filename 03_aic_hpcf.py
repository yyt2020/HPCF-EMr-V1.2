# -*- coding: utf-8 -*-
"""
Parallel AIC analysis of the stable point on the EM curve to estimate the
high-pass corner frequency (HPCF). Supports multiprocessing, a progress bar,
and process-safe plotting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import configparser
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend required for multiprocessing.

# ============================= Configuration =============================
config = configparser.ConfigParser()
with open('config.ini', 'r', encoding='utf-8') as f:
    config.read_file(f)

nn = float(config['AIC_Parameters']['nn'].split()[0])
C = float(config['AIC_Parameters']['C'].split()[0])
Earthquake_name = config['General_Parameters']['Earthquake_name'].split()[0]
root_dir = config['Paths']['root_dir'].split()[0]
aic_dir0 = config['Paths']['aic_dir'].split()[0]

indir_aic = os.path.join(root_dir, 'OUTPUT/02_em_curves')
outdir_aic = os.path.join(root_dir, 'OUTPUT/03_aic_hpcf_results')
aic_dir = os.path.join(root_dir, 'OUTPUT', aic_dir0)
out_filename4 = os.path.join(outdir_aic, f'{Earthquake_name}_em_hpcf.csv')

# Number of worker processes.
N_process = int(config['EM_Score_Parameters']['N_process'].split()[0])

# ======================== Globals used by workers ========================
os.makedirs(aic_dir, exist_ok=True)
os.makedirs(outdir_aic, exist_ok=True)

# Initialize the result file.
with open(out_filename4, 'w', encoding='utf-8') as f:
    f.write('Record,HPCF_Hz\n')

# ============================ AIC calculation ============================
def calculate_aic(data, nn, df):
    N = len(data)
    aic_values = np.zeros(N, dtype=float)
    for k in range(1, N - 1):
        var1 = np.var(data[:k]) + C
        var2 = np.var(data[k:]) + C
        aic_values[k] = k * np.log(var1) + (N - k) * np.log(var2) * nn
    k_min = int(np.argmin(aic_values))
    hp_freq_pick = (k_min + 1) * df
    return hp_freq_pick, aic_values

# ========================== Process-safe plotting ==========================
def plot_EM_and_AIC(EM_diff_norm, Frequency, aic_f, aic_curve, sta_name):
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 3), dpi=200)
    axes[0].plot(Frequency, EM_diff_norm, color='black', linewidth=1.0)
    axes[0].axhline(0, color='black', linestyle='--', linewidth=0.6)
    axes[0].axvline(aic_f, color='tab:blue', linestyle='-', linewidth=1.2)
    axes[0].set_ylabel('EM Slope Intensity')
    axes[0].grid(True, linestyle='--', linewidth=0.3, alpha=0.7)
    axes[0].legend([f'Pick = {aic_f:.3f} Hz'], loc='best', fontsize=9)

    axes[1].plot(Frequency, aic_curve, color='black', linewidth=1.0)
    axes[1].axvline(aic_f, color='tab:blue', linestyle='-', linewidth=1.2)
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('AIC')
    axes[1].grid(True, linestyle='--', linewidth=0.3, alpha=0.7)

    fig.suptitle("(b)", fontsize=14)
    fig.tight_layout()

    save_path = os.path.join(aic_dir, f'{sta_name}_em_aic_pick.png')
    fig.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close(fig)

# ========================== Single-record worker ==========================
def process_single_station(row):
    try:
        sta_name = row[0]
        df = float(row[1])
        EM = [float(x) for x in row[2:-1]]  # Exclude the potentially malformed last element.

        EM = np.array(EM, dtype=float)
        eps = 1e-12
        EM_diff = np.abs(np.log(np.abs((EM[:-1] + eps) / (EM[1:] + eps))))#
 
        EM_diff_norm = np.array(EM_diff, dtype=float)

        # Forward-order AIC calculation.
        hp_f, aic_forward = calculate_aic(EM_diff_norm, nn, df)

        # Frequency axis.
        N = len(EM_diff_norm)
        Frequency = np.arange(1, N + 1) * df

        # Optional process-safe diagnostic plot.
        #plot_EM_and_AIC(EM_diff_norm, Frequency, hp_f, aic_forward, sta_name)

        return sta_name, round(hp_f, 3)

    except Exception as e:
        print(f"[ERROR] Failed to process record {row[0]}: {e}")
        return row[0], -999.999  # Error marker.

# ============================= Read EM data =============================
def extract_rows_from_csv(file_path, skip_header_lines=1, encoding='latin-1'):
    rows = []
    with open(file_path, 'r', encoding=encoding, errors='replace') as f:
        for _ in range(skip_header_lines):
            next(f, None)
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(',', ' ').split()
            if len(parts) < 3:
                continue
            rows.append(parts)
    return rows

# ========================== Parallel entry point ==========================
def Subroutine4_AIC_fun_parallel():
    start_time = time.time()

    print(f"Starting parallel processing with {N_process} worker processes...")
    input_file = os.path.join(indir_aic, f'{Earthquake_name}_adjacent_EM_fixedRange_RAW.csv')
    rows = extract_rows_from_csv(input_file)

    results = []
    with ProcessPoolExecutor(max_workers=N_process) as executor:
        futures = [executor.submit(process_single_station, row) for row in rows]

        for future in tqdm(as_completed(futures), total=len(futures), desc="HPCF picking"):
            sta_name, hpcf = future.result()
            results.append((sta_name, hpcf))

    # Write the final CSV in completion order, matching the original implementation.
    with open(out_filename4, 'a', encoding='utf-8') as f:
        for sta_name, hpcf in results:
            f.write(f'{sta_name},{hpcf}\n')

    end_time = time.time()
    print(f"\nCompleted {len(results)} records.")
    print(f"Elapsed time: {end_time - start_time:.2f} s")
    print(f"Results saved to: {out_filename4}")
    print(
        "AIC diagnostic plotting remains disabled in the original workflow; "
        f"the reserved directory is: {aic_dir}"
    )

if __name__ == "__main__":
    Subroutine4_AIC_fun_parallel()
