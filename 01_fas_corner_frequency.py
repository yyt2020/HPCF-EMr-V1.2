# -*- coding: utf-8 -*-
"""
Stage 1: calculate the Fourier amplitude spectrum (FAS), apply
Konno-Ohmachi smoothing, analyze the log-log spectral slope, and estimate the
high-pass and low-pass corner frequencies (HPCF and LPCF).

Workflow
--------
1. Read config.ini and the station metadata CSV to obtain record/event pairs,
   worker count, input paths, and output paths.
2. For each record, apply a taper, calculate the FAS with an FFT, and project
   the smoothed spectrum onto 300 logarithmically spaced frequencies.
3. Identify the complete rising region R0 from positive-slope segments. Use
   its mean slope to estimate the HPCF (formerly f0/low_corner).
4. Search after R0 for the principal and fastest falling segments. The right
   endpoint of the fastest falling segment defines the LPCF.
5. Save FAS and derivative diagnostic plots with the HPCF/LPCF annotations.
6. Aggregate records by event and calculate HPCF_mean from values within the
   10th-90th percentile interval. Save the final table to
   FAS_HPCF_LPCF_results.csv.
"""

import os
import logging
import multiprocessing as mp
from functools import partial
from typing import Tuple, List, Optional
import configparser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from scipy.fft import fft, fftfreq
from hpcf_filter import taper

# Logging format.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Use a widely available Latin font because all labels are English.
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12})
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11

# ========================= Konno-Ohmachi smoothing =========================
try:
    from esi_core.gmprocess.waveform_processing.smoothing.konno_ohmachi import (
        konno_ohmachi_smooth,
    )
    _HAS_KO = True
except Exception:
    konno_ohmachi_smooth = None
    _HAS_KO = False

# ============================== Global settings ==============================
NKOFREQS: int = 300
MIN_FREQ: float = 0.01
MAX_FREQ: float = 100.0

SEG_CFG = dict(
    smooth_win=5,            
    min_freq=MIN_FREQ,       
    max_freq=MAX_FREQ,       
    low_freq_threshold=1.0   
)

# ============================== Record reader ==============================
def read_data_fun(file_path: str) -> Tuple[Optional[np.ndarray], Optional[float]]:
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


def calculate_fas(data: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray, int]:
    data = taper(data)
    n = len(data)
    n_fft = 2 ** int(np.ceil(np.log2(n)))
    data_padded = np.pad(data, (0, n_fft - n), mode='constant', constant_values=0)
    freq = fftfreq(n_fft, dt)
    fas = np.abs(fft(data_padded)) * dt
    return freq[: n_fft // 2], fas[: n_fft // 2], n_fft


def smooth_spectrum(
    spec: np.ndarray,
    freqs: np.ndarray,
    out_freqs: np.ndarray,
    smoothing_parameter: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    mask = (freqs > 0) & np.isfinite(spec) & (spec > 0)
    valid_freqs = freqs[mask]
    valid_spec = spec[mask]

    if not valid_freqs.size:
        logging.warning("No valid frequency samples; returning the original data.")
        return spec.copy(), freqs.copy()

    of = out_freqs
    if of[0] < valid_freqs[0] or of[-1] > valid_freqs[-1]:
        mask_of = (of >= valid_freqs[0]) & (of <= valid_freqs[-1])
        of_used = of[mask_of]
    else:
        mask_of = None
        of_used = of

    if of_used.size == 0:
        logging.warning(
            "The common frequency grid barely overlaps the valid range; "
            "returning the original data."
        )
        return spec.copy(), freqs.copy()

    spec_smooth_full = np.zeros_like(of, dtype=float)

    if _HAS_KO and (konno_ohmachi_smooth is not None):
        tmp = np.zeros_like(of_used, dtype=float)
        konno_ohmachi_smooth(
            valid_spec.astype(np.double),
            valid_freqs.astype(np.double),
            of_used.astype(np.double),
            tmp,
            smoothing_parameter,
        )
        if mask_of is None:
            spec_smooth_full = tmp
        else:
            spec_smooth_full[mask_of] = tmp
            if mask_of.any():
                first = np.where(mask_of)[0][0]
                last = np.where(mask_of)[0][-1]
                spec_smooth_full[:first] = tmp[0]
                spec_smooth_full[last + 1:] = tmp[-1]
    else:
        print('not install gmprocess')

    return spec_smooth_full, of


def compute_log_derivative(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    logx = np.log10(x)
    logy = np.log10(y)
    n = len(logx)
    slope = np.zeros(n, dtype=float)
    slope[1:-1] = (logy[2:] - logy[:-2]) / (logx[2:] - logx[:-2])
    slope[0] = (logy[1] - logy[0]) / (logx[1] - logx[0])
    slope[-1] = (logy[-1] - logy[-2]) / (logx[-1] - logx[-2])
    return slope


def moving_average(a: np.ndarray, win: int = 5) -> np.ndarray:
    if win <= 1:
        return a.copy()
    ker = np.ones(win, float) / win
    return np.convolve(a, ker, mode='same')


def find_segments(freqs: np.ndarray, slope: np.ndarray):
    signs = np.sign(slope)
    segments = []
    start = 0
    current_sign = signs[0]
    for i in range(1, len(slope)):
        if signs[i] != current_sign:
            if current_sign != 0:
                segments.append((start, i, current_sign))
            start = i
            current_sign = signs[i]
    if current_sign != 0:
        segments.append((start, len(slope), current_sign))
    return segments


def compute_fas_log_diff(spec: np.ndarray, start: int, end: int) -> float:
    if end - start < 1 or spec[start] <= 0 or spec[end - 1] <= 0:
        return 0.0
    return float(np.log10(spec[end - 1]) - np.log10(spec[start]))


def compute_log_freq_diff(freqs: np.ndarray, start: int, end: int) -> float:
    if end - start < 1 or freqs[start] <= 0:
        return 0.0
    return float(np.log10(freqs[end - 1]) - np.log10(freqs[start]))


def find_main_rising_segment(
    freqs: np.ndarray,
    slope: np.ndarray,
    spec: np.ndarray,
    cfg=SEG_CFG,
):
    s = moving_average(slope, cfg['smooth_win'])
    mask = (freqs >= cfg['min_freq']) & (freqs <= cfg['max_freq'])
    f = freqs[mask]
    s = s[mask]
    spec_m = spec[mask]

    default_ret = (np.nan, np.nan, np.nan, None, None, None, np.nan, [], None, None, np.nan, None, None, np.nan)
    
    if len(f) < 3:
        return default_ret

    f_peak = float(f[np.argmax(spec_m)])

    segments = find_segments(f, s)
    pos_segments = [seg for seg in segments if seg[2] > 0]
    if not pos_segments:
        return (np.nan, np.nan, np.nan, None, None, None, f_peak, segments, None, None, np.nan, None, None, np.nan)

    changes = [compute_fas_log_diff(spec_m, a, b) for a, b, _ in pos_segments]
    ry_idx = int(np.argmax(changes))
    ry_seg = pos_segments[ry_idx]
    ry_mid = np.sqrt(f[ry_seg[0]] * f[ry_seg[1] - 1])

    log_diffs = [compute_log_freq_diff(f, a, b) for a, b, _ in pos_segments]
    rx_idx = int(np.argmax(log_diffs))
    rx_seg = pos_segments[rx_idx]
    rx_mid = np.sqrt(f[rx_seg[0]] * f[rx_seg[1] - 1])

    if ry_mid > rx_mid:
        ry_seg, rx_seg = rx_seg, ry_seg
        ry_idx, rx_idx = rx_idx, ry_idx
        ry_mid, rx_mid = rx_mid, ry_mid

    r0_start_idx_raw = min(ry_seg[0], rx_seg[0])
    r0_end_idx_raw = max(ry_seg[1], rx_seg[1])

    f_r0_min = f[r0_start_idx_raw]
    f_r0_max = f[r0_end_idx_raw - 1]

    f_r0_max_new = max(f_r0_max, 1.0)

    r0_mask = (f >= f_r0_min) & (f <= f_r0_max_new)
    r0_idx = np.where(r0_mask)[0]

    if len(r0_idx) == 0:
        r0_start = r0_start_idx_raw
        r0_end = r0_end_idx_raw
    else:
        r0_start = int(r0_idx[0])
        r0_end = int(r0_idx[-1] + 1)

    r0_seg = (r0_start, r0_end)

    neg_segs = [
        seg for seg in segments
        if seg[2] < 0
        and r0_start <= seg[0]
        and seg[1] <= r0_end
        and np.all(f[seg[0]:seg[1]] < cfg['low_freq_threshold'])
    ]
    neg_segs = sorted(neg_segs, key=lambda x: f[x[0]])
    neg_changes = [compute_fas_log_diff(spec_m, a, b) for a, b, _ in neg_segs]

    threshold = max(changes[ry_idx], changes[rx_idx]) / 4.0
    valid_neg = [seg for seg, ch in zip(neg_segs, neg_changes) if -ch > threshold]

    dn_seg = None
    main_seg = r0_seg
    if valid_neg:
        dn_seg = max(valid_neg, key=lambda x: f[x[1] - 1])
        main_seg = (dn_seg[1], r0_end)

    ms, me = main_seg
    main_start_freq = float(f[ms])
    main_end_freq = float(f[me - 1])

    mslope = s[ms:me]
    avg_slope = float(np.mean(mslope[mslope > 0])) if np.any(mslope > 0) else np.nan
    if np.isnan(avg_slope):
        hpcf = np.nan
    else:
        idxs = np.where(s[ms:me] >= avg_slope / 2.0)[0]
        hpcf = float(f[ms + idxs[0]]) if len(idxs) else main_start_freq

    lpcf = np.nan
    main_falling_seg = None
    fastest_falling_seg = None
    avg_fall_slope = np.nan
    
    if r0_seg is not None:
        r0_end_idx = r0_seg[1]
        falling_segs_after_r0 = []
        for seg in segments:
            if seg[2] < 0: 
                adj_start = max(seg[0], r0_end_idx)
                if seg[1] > adj_start:
                    falling_segs_after_r0.append((adj_start, seg[1]))
                    
        if falling_segs_after_r0:
            main_falling_seg = max(falling_segs_after_r0, key=lambda x: x[1] - x[0])
            avg_fall_slope = np.mean(s[main_falling_seg[0]:main_falling_seg[1]])
            search_start = r0_end_idx
            search_slopes = s[search_start:]
            is_steep = search_slopes < avg_fall_slope
            steep_segs = []
            curr_start = -1
            for i, is_true in enumerate(is_steep):
                if is_true:
                    if curr_start == -1:
                        curr_start = i
                else:
                    if curr_start != -1:
                        steep_segs.append((curr_start, i))
                        curr_start = -1
            if curr_start != -1:
                steep_segs.append((curr_start, len(is_steep)))
                
            if steep_segs:
                fastest_seg_local = max(steep_segs, key=lambda x: x[1] - x[0])
                fastest_falling_seg = (search_start + fastest_seg_local[0], search_start + fastest_seg_local[1])
                lpcf_idx = fastest_falling_seg[1] - 1
                lpcf = float(f[lpcf_idx])

    return (
        main_start_freq,
        main_end_freq,
        hpcf,
        ry_seg,
        rx_seg,
        dn_seg,
        f_peak,
        segments,
        r0_seg,
        main_seg,
        lpcf,
        main_falling_seg,
        fastest_falling_seg,
        avg_fall_slope
    )


def draw_box(ax, f_start, f_end, label=None, color="#2ca02c", ypad=0.05, ls="--", lw=2.0, fill=False, alpha=0.2, hh=0.84):
    if not (np.isfinite(f_start) and np.isfinite(f_end) and f_end > f_start):
        return
    y_min, y_max = ax.get_ylim()
    H = (y_max - y_min) * (1 - 2 * ypad)
    Y0 = y_min + (y_max - y_min) * ypad
    rect = patches.Rectangle(
        (f_start, Y0),
        f_end - f_start,
        H,
        fill=fill,
        facecolor=(color if fill else "none"),
        alpha=(alpha if fill else 1.0),
        edgecolor=color,
        lw=lw,
        ls=ls,
        zorder=3,
    )
    ax.add_patch(rect)
    if label:
        ax.text(
            np.sqrt(f_start * f_end),
            Y0 + H * hh,
            label,
            color=color,
            fontsize=10,
            ha="center",
            va="top",
            zorder=4,
        )

def draw_vertical_line(ax, freq, label=None, color="red", ls="--", lw=1.5):
    if np.isfinite(freq):
        ax.axvline(freq, color=color, ls=ls, lw=lw)
        if label:
            ax.text(
                freq,
                ax.get_ylim()[1] * 0.95,
                label,
                color=color,
                fontsize=10,
                ha="left",
                va="top",
                rotation=90,
            )


def process_file(
    task_args: Tuple[str, str], 
    data_dir: str,
    fas_plot_dir: str,
    deriv_plot_dir: str,
    columns: list,
    ko_freqs: np.ndarray,
):
    
    target_name, eq_name = task_args
    potential_files = [target_name, target_name + ".dat"]
    
    filename = None
    filepath = None
    
    for f in potential_files:
        p = os.path.join(data_dir, f)
        if os.path.exists(p) and os.path.isfile(p):
            filename = f
            filepath = p
            break
            
    if filepath is None:
        return None

    try:
        time_history, dt = read_data_fun(filepath)
        if time_history is None or dt is None:
            return None

        freq, fas, nfft = calculate_fas(time_history, dt)
        fas_smooth, ko_freqs_used = smooth_spectrum(fas, freq, ko_freqs)

        EPS = 1e-30
        spec = np.maximum(fas_smooth, EPS)
        slope = compute_log_derivative(spec, ko_freqs_used)

        (
            main_start_freq,
            main_end_freq,
            hpcf,
            ry_seg,
            rx_seg,
            dn_seg,
            f_peak,
            segments,
            r0_seg,
            main_seg,
            lpcf,
            main_falling_seg,
            fastest_falling_seg,
            avg_fall_slope
        ) = find_main_rising_segment(ko_freqs_used, slope, spec, SEG_CFG)
        
        plt.figure(figsize=(10, 6))
        plt.loglog(freq[freq > 0], fas[freq > 0], label='FAS', color='blue', linewidth=1.0)
        plt.loglog(
            ko_freqs_used,
            fas_smooth,
            label='Smoothed FAS',
            color='orange',
            linewidth=2.0,
        )

        freq_line = np.array([0.1, 10.0])
        fas_line = 0.01 * freq_line ** 2
        plt.loglog(freq_line, fas_line, color='green', linestyle='--', linewidth=1.5, label='Slope = 2')
        plt.axvline(x=hpcf, color='red', linestyle='--', linewidth=1.5, label="HPCF")
        
        if np.isfinite(lpcf):
            plt.axvline(x=lpcf, color='purple', linestyle='-.', linewidth=1.5, label="LPCF")

        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Fourier amplitude spectrum (gal)')
        plt.title(f'Record: {filename} (event: {eq_name})')
        plt.grid(True, which="both", ls="--", alpha=0.6)
        plt.legend()
        fas_png = os.path.join(fas_plot_dir, f"{os.path.splitext(filename)[0]}_FAS.png")
        plt.savefig(fas_png, dpi=300, bbox_inches='tight')
        plt.close()
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogx(ko_freqs_used, slope, color="blue", lw=2, label="Spectral slope")
        ax.axhline(2, color="black", ls="--", lw=1.2, label="Slope = 2")
        ax.axhline(1, color="gray", ls="--", lw=1)
        ax.axhline(3, color="gray", ls="--", lw=1)
        ax.axhline(0, color="green", ls="-.", lw=1.2, label="Slope = 0")

        if np.isfinite(avg_fall_slope):
            ax.axhline(
                avg_fall_slope,
                color="purple",
                ls="--",
                lw=1.5,
                label=f"Mean principal-fall slope ({avg_fall_slope:.2f})",
            )

        mask = (ko_freqs_used >= SEG_CFG['min_freq']) & (ko_freqs_used <= SEG_CFG['max_freq'])
        f_masked = ko_freqs_used[mask]

        if ry_seg:
            draw_box(ax, f_masked[ry_seg[0]], f_masked[ry_seg[1] - 1], "Ry", color="#ff7f0e", alpha=0.1, fill=True)
        if rx_seg:
            draw_box(ax, f_masked[rx_seg[0]], f_masked[rx_seg[1] - 1], "Rx", color="#2ca02c", alpha=0.1, fill=True)
        if dn_seg:
            draw_box(ax, f_masked[dn_seg[0]], f_masked[dn_seg[1] - 1], "DN", color="#d62728", alpha=0.1, fill=True)
        if r0_seg:
            draw_box(ax, f_masked[r0_seg[0]], f_masked[r0_seg[1] - 1], "R0", color="gray", ls="-", lw=2, alpha=0.5, hh=0.92)
        if main_seg:
            draw_box(ax, f_masked[main_seg[0]], f_masked[main_seg[1] - 1], "Rf", color="black", ls="--", lw=2, alpha=0.5, hh=0.92)

        if main_falling_seg:
            draw_box(ax, f_masked[main_falling_seg[0]], f_masked[main_falling_seg[1] - 1], "Main Fall", color="mediumpurple", ls=":", lw=1.5, alpha=0.15, fill=True)
        if fastest_falling_seg:
            draw_box(ax, f_masked[fastest_falling_seg[0]], f_masked[fastest_falling_seg[1] - 1], "Fastest Fall", color="purple", ls="-", lw=2, alpha=0.25, fill=True)

        draw_vertical_line(ax, hpcf, label="HPCF", color="red")
        if np.isfinite(lpcf):
            draw_vertical_line(ax, lpcf, label="LPCF", color="purple")

        ax.set_title(str(filename), fontsize=13)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Slope dlogFAS/dlogf")
        ax.grid(True, which="both", ls="--", alpha=0.6)
        ax.legend(loc="best")

        deriv_png = os.path.join(
            deriv_plot_dir,
            f"{os.path.splitext(filename)[0]}_derivative.png",
        )
        plt.savefig(deriv_png, dpi=300, bbox_inches="tight")
        plt.close(fig)

        row_values = [filename, eq_name, f_peak, hpcf, lpcf] + fas_smooth.tolist()
        return pd.Series(row_values, index=columns)

    except Exception as e:
        logging.error(f"Failed to process {filename}: {e}")
        return None


def main():
    config = configparser.ConfigParser()
    if not os.path.exists('config.ini'):
        print("Error: config.ini not found.")
        return
        
    with open('config.ini', 'r', encoding='utf-8') as f:
        config.read_file(f)

    csv_file = str(config['Paths']['station_metadata'].split()[0])
    num_cores = int(config['EM_Score_Parameters']['N_process'].split()[0])
    data_dir = str(config['Paths']['in_dir'].split()[0])
    output_dir = str(config['FAS_Parameters']['FAS_outdir'].split()[0])
    
    fas_plot_dir = os.path.join(output_dir, "FAS_plots")
    deriv_plot_dir = os.path.join(output_dir, "FAS_derivative_plot")
    smooth_csv = os.path.join(output_dir, "FAS_smoothed.csv")
    result_csv = os.path.join(output_dir, "FAS_HPCF_LPCF_results.csv")
    
    os.makedirs(fas_plot_dir, exist_ok=True)
    os.makedirs(deriv_plot_dir, exist_ok=True)
    
    try:
        df = pd.read_csv(csv_file, encoding='latin1') 
        record_names = df.iloc[:, 4].astype(str).str.strip().tolist()
        earthquake_names = df.iloc[:, 6].astype(str).str.strip().tolist()
        tasks = list(zip(record_names, earthquake_names))
        print(f"Loaded {len(tasks)} record tasks from the metadata CSV.")
    except Exception as e:
        print(f"[ERROR] Failed to read the metadata CSV: {e}")
        return

    ko_freqs = np.logspace(np.log10(MIN_FREQ), np.log10(MAX_FREQ), NKOFREQS)
    columns = ["Record", "Earthquake_Name", "f_peak", "HPCF", "LPCF"] + [f"{f:.4f}Hz" for f in ko_freqs]

    with mp.Pool(num_cores) as pool:
        process_func = partial(
            process_file,
            data_dir=data_dir,
            fas_plot_dir=fas_plot_dir,
            deriv_plot_dir=deriv_plot_dir,
            columns=columns,
            ko_freqs=ko_freqs,
        )
        results = pool.map(process_func, tasks)

    results = [r for r in results if r is not None]
    if results:
        smooth_df = pd.concat(results, axis=1).T
        smooth_df.columns = columns
        
        smooth_df['HPCF'] = pd.to_numeric(smooth_df['HPCF'], errors='coerce')
        smooth_df['LPCF'] = pd.to_numeric(smooth_df['LPCF'], errors='coerce')
        
        smooth_df.to_csv(smooth_csv, index=False, encoding='utf-8-sig')
        print(f"Smoothed FAS data saved to: {smooth_csv}")

        def calc_p10_p90_mean(series):
            valid_data = series.dropna()
            if valid_data.empty:
                return np.nan
            p10 = valid_data.quantile(0.10)
            p90 = valid_data.quantile(0.90)
            mask = (valid_data >= p10) & (valid_data <= p90)
            filtered = valid_data[mask]
            return filtered.mean() if not filtered.empty else np.nan

        smooth_df['HPCF_mean'] = smooth_df.groupby('Earthquake_Name')['HPCF'].transform(calc_p10_p90_mean)
        
        final_df = smooth_df[["Record", "HPCF", "HPCF_mean", "LPCF"]]
        final_df.to_csv(result_csv, index=False, encoding="utf-8-sig")

        print(f"Spectral-slope plots saved to: {deriv_plot_dir}")
        print(f"HPCF/LPCF results saved to: {result_csv}")
    else:
        print("[WARNING] No records were processed; check that input filenames match the metadata CSV.")


if __name__ == '__main__':
    main()
