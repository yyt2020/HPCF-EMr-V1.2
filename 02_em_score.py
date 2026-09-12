# -*- coding: utf-8 -*-
"""
Compute adjacent-record envelope-misfit (EM) scores over a high-pass frequency
grid. The frequency grid is derived from the stage-1 HPCF mean, and optional
dynamic downsampling is controlled by config.ini. The computational procedure
is unchanged from the reference implementation used for this study.
"""
import os
import time
import logging
import configparser
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
from multiprocessing import Pool, Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from obspy.signal.tf_misfit import tem

# The filtering helper is distributed with this package.
try:
    from hpcf_filter import timehistory_process
except ImportError:
    logging.error("The hpcf_filter module was not found in the working directory.")
    raise

# Logging format.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ======================
# Read configuration.
# ======================
config = configparser.ConfigParser()
if not os.path.exists('config.ini'):
    raise FileNotFoundError("config.ini was not found in the working directory.")

with open('config.ini', 'r', encoding='utf-8') as f:
    config.read_file(f)

# EM parameters.
w0 = float(config['EM_Score_Parameters']['w0'].split()[0])
total = int(config['EM_Score_Parameters']['total'].split()[0])
N_process = int(config['EM_Score_Parameters']['N_process'].split()[0])
fmax_tem = float(config['EM_Score_Parameters']['fmax'].split()[0])

# Downsampling control (default: True).
try:
    need_downsample_str = config['EM_Score_Parameters']['need_downsample'].split()[0]
    if need_downsample_str.lower() in ['true', '1', 'yes', 'on']:
        need_downsample = True
    else:
        need_downsample = False
except KeyError:
    need_downsample = True  # Fallback value.
    logging.info("need_downsample is absent from config.ini; using True.")

# Path parameters.
Earthquake_name = str(config['General_Parameters']['Earthquake_name'].split()[0])
root_dir = str(config['Paths']['root_dir'].split()[0])
in_dir = str(config['Paths']['in_dir'].split()[0])  # Raw waveform directory.
FAS_output_dir = str(config['FAS_Parameters']['FAS_outdir'].split()[0])  # Stage-1 output directory.

# Output settings.
outdir_em = os.path.join(root_dir, 'OUTPUT/02_em_curves')
os.makedirs(outdir_em, exist_ok=True)
out_adjacent = os.path.join(outdir_em, f'{Earthquake_name}_adjacent_EM_fixedRange_RAW.csv')

# Stage-1 result file.
input_csv_path = os.path.join(FAS_output_dir, "FAS_HPCF_LPCF_results.csv")

# Process lock used when appending output rows.
file_lock = Lock()


# ======================
# Robust CSV reader.
# ======================
def _read_csv_robust(path):
    """Read a CSV file by trying several common encodings."""
    for enc in ['utf-8-sig', 'utf-8', 'gb18030', 'gbk', 'latin-1']:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path)


# ======================
# Initialize the record list and HPCF lookup.
# ======================
def initialize_data_map():
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Input file not found: {input_csv_path}")

    df = _read_csv_robust(input_csv_path)

    # Normalize stage-1 column names by removing surrounding whitespace.
    df.columns = [c.strip() for c in df.columns]

    if 'Record' not in df.columns or 'HPCF_mean' not in df.columns:
        raise ValueError(
            f"Input file {input_csv_path} lacks required columns "
            "'Record' or 'HPCF_mean'."
        )

    target_files_list = []
    local_map = {}

    logging.info(f"Reading and matching {len(df)} rows from the input CSV...")

    for _, row in df.iterrows():
        record_name = str(row['Record']).strip()
        try:
            f0_mean_val = float(row['HPCF_mean'])
        except (ValueError, TypeError):
            continue  # Skip invalid values.

        # Build the corresponding physical record path.
        record = os.path.normpath(os.path.join(in_dir, record_name))

        target_files_list.append(record)
        local_map[record] = f0_mean_val

    # Remove duplicate paths.
    target_files_list = sorted(list(set(target_files_list)))
    logging.info(f"Matched {len(target_files_list)} record files.")
    logging.info(f"Dynamic downsampling enabled: {need_downsample}")

    return target_files_list, local_map


# Perform initialization.
try:
    target_files, path_fasmean_map = initialize_data_map()
except Exception as e:
    logging.error(f"Initialization failed: {e}")
    target_files = []
    path_fasmean_map = {}


# ======================
# Utility functions.
# ======================

# ============================== Record reader ==============================
def read_data_fun(file_path: str) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """Read a China Strong Motion Network record and return data and dt."""
    try:
        with open(file_path, "r", encoding='utf-8') as f:
            lines = f.readlines()
        base = os.path.basename(file_path).lower()
        if base.endswith('.dat'):
            dt = float(lines[11].split()[-2])
            vals: List[float] = []
            for line in lines[16:]:
                vals.extend(float(x) for x in line.split() if x.strip())
        else:
            dt = float(lines[41].split()[-2])
            vals = []
            for line in lines[49:]:
                vals.extend(float(x) for x in line.split() if x.strip())
        if not vals:
            raise ValueError("The time series is empty.")
        return np.asarray(vals, dtype=float), float(dt)
    except Exception as e:
        logging.error(f"Failed to read {file_path}: {e}")
        return None, None



def rms(x):
    """Calculate the root-mean-square value."""
    x = np.asarray(x, dtype=float)
    return np.nan if x.size == 0 else float(np.sqrt(np.mean(np.square(x))))


def normalize_by_pgd(dis):
    """Displacement normalization placeholder (currently the identity)."""
    return dis


def _filter_displacement(args):
    """Filter and downsample; return (fHP, dis_ds, dt_ds, err)."""
    fHP, acc_raw, dt, npts, decimate_factor = args
    try:
        # Apply the distributed time-history processing function.
        _, _, dis = timehistory_process(
                                        timehistory=acc_raw,
                                        dt=dt,
                                        filter_sort=1,  # High-pass.
                                        order=5,
                                        fmin=float(fHP),
                                        fmax=fmax_tem,
                                        Tp=0,
                                        is_pad=1,
                                        save_pad=0,
                                        filter_domain='freq_noncausal'
                                    )

        if len(dis) != npts:
            return (fHP, None, None, f"Length mismatch: {len(dis)} vs {npts}")

        # Downsample when requested.
        if decimate_factor > 1:
            dis_ds = dis[::decimate_factor].copy()
            dt_ds = dt * decimate_factor
        else:
            dis_ds = dis
            dt_ds = dt

        return (fHP, normalize_by_pgd(dis_ds), dt_ds, None)
    except Exception as e:
        return (fHP, None, None, str(e))


# ======================
# Frequency-grid construction.
# ======================
def build_freq_grid_from_fasmean(file_path, dt_after_decimate, nyquist_margin=0.99):
    """Build the frequency grid from the HPCF mean mapped to a record path."""
    norm_path = os.path.normpath(file_path)
    fas_mean = path_fasmean_map.get(norm_path, None)

    if fas_mean is None:
        logging.warning(f"No HPCF mean mapping found for {os.path.basename(file_path)}.")
        return None, None
    if not np.isfinite(fas_mean) or fas_mean <= 0:
        return None, None

    nyq = 0.5 / dt_after_decimate
    f_max_target = max(5.0 * fas_mean, 0.1)  # Search up to five times the HPCF mean.
    f_max_allow = min(f_max_target, nyq * nyquist_margin, fmax_tem)

    if f_max_allow <= 0:
        return None, None

    step = round(f_max_allow / 100.0, 4)
    if step <= 0:
        return None, None

    freqs = np.round(step * np.arange(1, 101), 4)
    freqs = np.unique(freqs)
    return freqs, step


# ======================
# Core processing function.
# ======================
def process_file_adjacent(args):
    """Process one acceleration record."""
    file_path, out_adj_file = args
    try:
        acc_raw, dt = read_data_fun(file_path)
        if acc_raw is None:
            return

        sta_full = os.path.splitext(os.path.basename(file_path))[0]
        npts = acc_raw.size

        # ==========================================
        # Dynamic downsampling control.
        # ==========================================
        decimate_factor = 1
        if need_downsample:
            # Select the largest factor that keeps dt_ds <= 0.05 s
            # (Nyquist frequency >= 10 Hz).
            target_dt_ds = 0.05
            if dt < target_dt_ds:
                decimate_factor = int(target_dt_ds / dt)
                if decimate_factor < 1:
                    decimate_factor = 1

        dt_ds_trial = dt * decimate_factor

        # Build the frequency grid.
        freqs, step = build_freq_grid_from_fasmean(file_path, dt_after_decimate=dt_ds_trial)
        if freqs is None or len(freqs) < 2:
            logging.warning(f"Frequency-grid generation failed for {sta_full}; skipping.")
            return

        # Filter the frequency candidates in parallel.
        dis_dict = {}
        tasks = [(float(fHP), acc_raw, dt, npts, decimate_factor) for fHP in freqs]
        # The outer process pool is combined with up to six filtering threads
        # per record, as in the original implementation.
        with ThreadPoolExecutor(max_workers=min(6, len(tasks))) as ex:
            futures = {ex.submit(_filter_displacement, t): t[0] for t in tasks}
            for fut in as_completed(futures):
                fHP = futures[fut]
                try:
                    f_return, dis_ds, dt_ds, err = fut.result()
                except Exception as e:
                    logging.warning(f"Internal filtering error for {sta_full}, fHP={fHP:.6f}: {e}")
                    continue
                if err is not None or dis_ds is None or dt_ds is None:
                    logging.warning(f"Filtering failed for {sta_full}, fHP={fHP:.6f}: {err}")
                    continue
                dis_dict[float(f_return)] = (dis_ds, dt_ds)

        f_sorted = np.array(sorted(dis_dict.keys(), key=float), dtype=float)
        if f_sorted.size < 2:
            logging.error(f"{sta_full} has only {f_sorted.size} valid frequencies; skipping.")
            return

        # Calculate EM between adjacent filtered waveforms (gap_k = 1).
        gap_k = 1
        em_fixed_gap = []
        for i in range(0, len(f_sorted) - gap_k):
            f1 = float(f_sorted[i])
            f2 = float(f_sorted[i + gap_k])
            d1, dt1 = dis_dict.get(f1, (None, None))
            d2, dt2 = dis_dict.get(f2, (None, None))

            if d1 is None or d2 is None:
                em_fixed_gap.append(np.nan)
                continue

            m = min(len(d1), len(d2))
            if m < 2:
                em_fixed_gap.append(np.nan)
                continue

            d1c = d1[:m]
            d2c = d2[:m]
            dt_ds = 0.5 * (dt1 + dt2)
            nyq_ds = 0.5 / dt_ds
            fmax_use = min(fmax_tem, nyq_ds * 0.99)
            fmin_use = max(f1, f2)

            if not (fmin_use < fmax_use):
                em_fixed_gap.append(np.nan)
                continue

            try:
                TEM = tem(d1c, d2c, dt=dt_ds, fmin=fmin_use, fmax=fmax_use,
                          w0=w0, norm='local', st2_isref=True)
                em_val = rms(TEM)
                em_fixed_gap.append(em_val if np.isfinite(em_val) else np.nan)
            except Exception as e:
                logging.warning(f"TEM calculation failed for {sta_full}, {f1:.6f}-{f2:.6f}: {e}")
                em_fixed_gap.append(np.nan)

        # Format and append one output row.
        em_strs = [f"{v:.6f}" if np.isfinite(v) else "NaN" for v in em_fixed_gap]
        row_fields = [sta_full, f"{step:.6f}"] + em_strs
        line = ",".join(row_fields) + "\n"

        with file_lock:
            with open(out_adjacent, 'a', encoding='utf-8') as fa:
                fa.write(line)
        logging.info(
            f"Completed {sta_full} | df={step:.6f} Hz | "
            f"frequencies={len(f_sorted)} -> EM values={len(em_fixed_gap)} | "
            f"downsampling factor={decimate_factor}"
        )
    except Exception as e:
        logging.error(f"Failed to process {file_path}: {e}")


# ======================
# Entry point.
# ======================
def main():
    start = time.time()

    # Write the header.
    with open(out_adjacent, 'w', encoding='utf-8') as f:
        f.write("Record,df,EM_values...\n")

    files = target_files
    if not files:
        logging.error(f"No records to process. Check {input_csv_path} and {in_dir}.")
        return

    N = len(files)
    total_new = N if total == 0 else min(total, N)  # Process all records when total = 0.
    num_cores = min(max(1, N_process), 100)

    logging.info(f"Input CSV: {input_csv_path}")
    logging.info(f"Matched records: {N}; scheduled: {total_new}; processes: {num_cores}")

    args = [(file, out_adjacent) for file in files[:total_new]]

    with Pool(processes=num_cores) as pool:
        pool.map(process_file_adjacent, args)

    logging.info(f"Elapsed time: {time.time() - start:.2f} s")
    logging.info(f"Output written to: {out_adjacent}")


if __name__ == "__main__":
    main()
