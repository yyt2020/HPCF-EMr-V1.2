# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 22:09:30 2025
@author: YYT

Extended features:
- The filter_domain parameter supports four filtering modes:
    1. 'time_causal'     -> causal time-domain filtering (sosfilt)
    2. 'time_noncausal'  -> noncausal time-domain filtering (sosfiltfilt, zero phase)
    3. 'freq_noncausal'  -> noncausal frequency-domain filtering (FFT x |H|^2, zero phase)
    4. 'freq_causal'     -> causal frequency-domain filtering (FFT x H, phase retained)
- The frequency-domain filters follow the MATLAB FR.m formulation and construct
  the analog Butterworth frequency response from its poles.
"""

import numpy as np
from scipy import signal
import scipy.integrate as spi
import math
from obspy.signal.detrend import spline
from scipy.integrate import cumulative_trapezoid


def taper(x, tb=5, te=5):
    """
    Apply a cosine taper to reduce filtering edge effects.
    tb: percentage tapered at the beginning (default: 5%).
    te: percentage tapered at the end (default: 5%).
    """
    x = np.array(x)
    n = len(x)
    n_front = int(math.ceil((n * tb) / 100.0))
    for i in range(1, n_front + 1):
        arg = np.pi * (i - 1) / n_front + np.pi
        x[i - 1] *= (1 + np.cos(arg)) / 2
    n_back = int(math.ceil((n * te) / 100.0))
    for i in range(1, n_back + 1):
        arg = np.pi * (i - 1) / n_back + np.pi
        x[n - i] *= (1 + np.cos(arg)) / 2
    return x


def add_zero(timehistory, dt, n, fc):
    """
    Pad both ends of the signal with zeros to reduce filtering boundary
    distortion (Boore, 2005).
    T_pad = 1.5 * n / fc
    """
    T_pad = 1.5 * n / fc
    pad_len = int(round(T_pad / dt))
    zero_pad = np.zeros(pad_len)
    timehistory_added = np.concatenate((zero_pad, timehistory, zero_pad))
    return timehistory_added, pad_len


def highpass_filter(value_list, fs, lowcut_freq, order, causal=False):
    """Apply a numerically stable SOS-form Butterworth high-pass filter."""
    wn = lowcut_freq / (fs / 2)
    sos = signal.butter(order, wn, "highpass", output='sos')
    if causal:
        filtered_signal = signal.sosfilt(sos, value_list)
    else:
        filtered_signal = signal.sosfiltfilt(sos, value_list)
    return filtered_signal


def lowpass_filter(signal_data, fs, highcut_freq, order, causal=False):
    """Apply an SOS-form Butterworth low-pass filter."""
    wn = highcut_freq / (fs / 2)
    sos = signal.butter(order, wn, btype='lowpass', output='sos')
    if causal:
        filtered_signal = signal.sosfilt(sos, signal_data)
    else:
        filtered_signal = signal.sosfiltfilt(sos, signal_data)
    return filtered_signal


def bandpass_filter(signal_data, fs, lowcut_freq, highcut_freq, order, causal=False):
    """Apply an SOS-form Butterworth band-pass filter."""
    low = lowcut_freq / (fs / 2)
    high = highcut_freq / (fs / 2)
    sos = signal.butter(order, [low, high], btype='bandpass', output='sos')
    if causal:
        filtered_signal = signal.sosfilt(sos, signal_data)
    else:
        filtered_signal = signal.sosfiltfilt(sos, signal_data)
    return filtered_signal


# ====================== Noncausal frequency-domain filter ======================
def freq_domain_filter(signal_data, fs, filter_type, lowcut_freq=None, highcut_freq=None, order=5):
    """
    Apply a noncausal, zero-phase Butterworth filter in the frequency domain.
    H(f) is constructed from the analog prototype (s-domain) poles exactly as
    in MATLAB FR.m. The operation is rFFT -> multiply by |H(f)|^2 -> irFFT.
    Adequate external padding/windowing is required to reduce edge artifacts.
    """
    x = np.asarray(signal_data, dtype=float)
    Nsig = len(x)
    if Nsig == 0:
        return x

    if filter_type not in ('highpass', 'lowpass', 'bandpass'):
        raise ValueError("filter_type must be 'highpass', 'lowpass', or 'bandpass'.")
    if order <= 0:
        raise ValueError("order must be a positive integer.")

    nyq = fs / 2.0
    if filter_type == 'lowpass':
        if highcut_freq is None or not (0 < highcut_freq < nyq):
            raise ValueError("Low-pass filtering requires 0 < highcut_freq < Nyquist.")
    elif filter_type == 'highpass':
        if lowcut_freq is None or not (0 < lowcut_freq < nyq):
            raise ValueError("High-pass filtering requires 0 < lowcut_freq < Nyquist.")
    else:
        if (lowcut_freq is None) or (highcut_freq is None):
            raise ValueError("Band-pass filtering requires both lowcut_freq and highcut_freq.")
        if not (0 < lowcut_freq < highcut_freq < nyq):
            raise ValueError("The cutoffs must satisfy 0 < lowcut_freq < highcut_freq < Nyquist.")

    freqs = np.fft.rfftfreq(Nsig, d=1.0/fs)  # One-sided frequency grid.

    def _butterworth_FRm_H(f_nonneg, fc, N, kind, eps=1e-30):
        # FR.m poles.
        k = np.arange(1, N + 1)
        poles = np.exp(1j * np.pi * (0.5 + (2 * k - 1) / (2 * N)))
        if kind == 'lowpass':
            S0 = 1j * (f_nonneg / fc)
        elif kind == 'highpass':
            S0 = 1.0 / (1j * np.maximum(f_nonneg, eps) / fc)
        else:
            raise ValueError("kind must be 'lowpass' or 'highpass'.")
        T = np.ones_like(S0, dtype=complex)
        for pk in poles:
            T *= (S0 - pk)
        return 1.0 / T

    if filter_type == 'lowpass':
        H = _butterworth_FRm_H(freqs, highcut_freq, order, 'lowpass')
    elif filter_type == 'highpass':
        H = _butterworth_FRm_H(freqs, lowcut_freq, order, 'highpass')
    else:
        H_hp = _butterworth_FRm_H(freqs, lowcut_freq, order, 'highpass')
        H_lp = _butterworth_FRm_H(freqs, highcut_freq, order, 'lowpass')
        H = H_hp * H_lp

    H0 = np.abs(H) ** 2  # Zero phase.
    X = np.fft.rfft(x)
    Y = X * H0
    y = np.fft.irfft(Y, n=Nsig)
    return y


# ======================== Causal frequency-domain filter ========================
def freq_domain_filter_causal(signal_data, fs, filter_type, lowcut_freq=None, highcut_freq=None, order=5):
    """
    Apply a causal Butterworth filter in the frequency domain while retaining
    phase. H(f) is constructed from the FR.m s-domain analog prototype. Unlike
    the zero-phase form, this routine directly uses H(f), not |H|^2. The
    operation is rFFT -> multiply by H(f) -> irFFT. External zero padding and
    windowing are recommended to reduce circular-convolution artifacts.
    """
    x = np.asarray(signal_data, dtype=float)
    Nsig = len(x)
    if Nsig == 0:
        return x

    if filter_type not in ('highpass', 'lowpass', 'bandpass'):
        raise ValueError("filter_type must be 'highpass', 'lowpass', or 'bandpass'.")
    if order <= 0:
        raise ValueError("order must be a positive integer.")

    nyq = fs / 2.0
    if filter_type == 'lowpass':
        if highcut_freq is None or not (0 < highcut_freq < nyq):
            raise ValueError("Low-pass filtering requires 0 < highcut_freq < Nyquist.")
    elif filter_type == 'highpass':
        if lowcut_freq is None or not (0 < lowcut_freq < nyq):
            raise ValueError("High-pass filtering requires 0 < lowcut_freq < Nyquist.")
    else:
        if (lowcut_freq is None) or (highcut_freq is None):
            raise ValueError("Band-pass filtering requires both lowcut_freq and highcut_freq.")
        if not (0 < lowcut_freq < highcut_freq < nyq):
            raise ValueError("The cutoffs must satisfy 0 < lowcut_freq < highcut_freq < Nyquist.")

    freqs = np.fft.rfftfreq(Nsig, d=1.0/fs)  # One-sided frequency grid.

    def _butterworth_FRm_H(f_nonneg, fc, N, kind, eps=1e-30):
        # FR.m poles.
        k = np.arange(1, N + 1)
        poles = np.exp(1j * np.pi * (0.5 + (2 * k - 1) / (2 * N)))
        if kind == 'lowpass':
            S0 = 1j * (f_nonneg / fc)
        elif kind == 'highpass':
            S0 = 1.0 / (1j * np.maximum(f_nonneg, eps) / fc)
        else:
            raise ValueError("kind must be 'lowpass' or 'highpass'.")
        T = np.ones_like(S0, dtype=complex)
        for pk in poles:
            T *= (S0 - pk)
        return 1.0 / T

    if filter_type == 'lowpass':
        H = _butterworth_FRm_H(freqs, highcut_freq, order, 'lowpass')
    elif filter_type == 'highpass':
        H = _butterworth_FRm_H(freqs, lowcut_freq, order, 'highpass')
    else:
        H_hp = _butterworth_FRm_H(freqs, lowcut_freq, order, 'highpass')
        H_lp = _butterworth_FRm_H(freqs, highcut_freq, order, 'lowpass')
        H = H_hp * H_lp

    X = np.fft.rfft(x)
    Y = X * H           # Use complex H directly, not |H|^2.
    y = np.fft.irfft(Y, n=Nsig)
    return y


# ===============================================================================


def timehistory_process(timehistory, dt, filter_sort, order, fmin, fmax, Tp,
                       is_pad=0, save_pad=0, filter_domain=None):
    """
    Main processing function: acceleration -> filtering -> integration to
    velocity and displacement.

    Added parameter:
    - filter_domain: filtering mode
        'time_causal'     -> causal time-domain filter (sosfilt)
        'time_noncausal'  -> noncausal time-domain filter (sosfiltfilt; default)
        'freq_noncausal'  -> noncausal frequency-domain filter (FFT x |H|^2)
        'freq_causal'     -> causal frequency-domain filter (FFT x H)

    filter_sort:
        1 -> high-pass, 2 -> band-pass, 3 -> low-pass
    """
    acc = np.array(timehistory)

    # 1. Remove the mean.
    if Tp > 5:
        idx_p = int(Tp / dt)
        idx_mean_end = int(0.9 * idx_p)
        if idx_mean_end > 0:
            mean_pre = np.mean(acc[:idx_mean_end])
            acc -= mean_pre
    else:
        acc -= np.mean(acc)


    # 3. Select the cutoff frequency used to determine zero padding.
    fc = fmin

    # 4. Zero-pad to reduce boundary effects.
    pad_len = 0
    if is_pad == 1:  # Includes freq_noncausal and freq_causal modes.
        acc = taper(acc)
        acc_padded, pad_len = add_zero(acc, dt, order, fc)
    else:
        acc_padded = acc.copy()

    # 5. Apply the selected filter.
    fs = 1.0 / dt

    if filter_domain == 'time_causal':
        # Causal time-domain filtering.
        if filter_sort == 1:
            acc_filtered = highpass_filter(acc_padded, fs, fmin, order, causal=True)
        elif filter_sort == 2:
            acc_filtered = bandpass_filter(acc_padded, fs, fmin, fmax, order, causal=True)
        elif filter_sort == 3:
            acc_filtered = lowpass_filter(acc_padded, fs, fmax, order, causal=True)

    elif filter_domain == 'time_noncausal':
        # Noncausal time-domain filtering (default).
        if filter_sort == 1:
            acc_filtered = highpass_filter(acc_padded, fs, fmin, order, causal=False)
        elif filter_sort == 2:
            acc_filtered = bandpass_filter(acc_padded, fs, fmin, fmax, order, causal=False)
        elif filter_sort == 3:
            acc_filtered = lowpass_filter(acc_padded, fs, fmax, order, causal=False)

    elif filter_domain == 'freq_noncausal':
        # Noncausal frequency-domain filtering (zero phase).
        if filter_sort == 1:
            acc_filtered = freq_domain_filter(acc_padded, fs, 'highpass', lowcut_freq=fmin, order=order)
        elif filter_sort == 2:
            acc_filtered = freq_domain_filter(acc_padded, fs, 'bandpass', lowcut_freq=fmin, highcut_freq=fmax, order=order)
        elif filter_sort == 3:
            acc_filtered = freq_domain_filter(acc_padded, fs, 'lowpass', highcut_freq=fmax, order=order)

    elif filter_domain == 'freq_causal':
        # Causal frequency-domain filtering (phase retained).
        if filter_sort == 1:
            acc_filtered = freq_domain_filter_causal(acc_padded, fs, 'highpass', lowcut_freq=fmin, order=order)
        elif filter_sort == 2:
            acc_filtered = freq_domain_filter_causal(acc_padded, fs, 'bandpass', lowcut_freq=fmin, highcut_freq=fmax, order=order)
        elif filter_sort == 3:
            acc_filtered = freq_domain_filter_causal(acc_padded, fs, 'lowpass', highcut_freq=fmax, order=order)
    else:
        raise ValueError(
            "filter_domain must be 'time_causal', 'time_noncausal', "
            "'freq_noncausal', or 'freq_causal'"
        )

    # 6. Build the time axis and integrate.
    t_padded = np.linspace(0, (len(acc_filtered) - 1) * dt, len(acc_filtered))
    vel = cumulative_trapezoid(acc_filtered, t_padded, initial=0)
    dis = cumulative_trapezoid(vel, t_padded, initial=0)

    # 7. Remove the padded samples when requested.
    if save_pad == 0:
        original_len = len(timehistory)
        start_idx = pad_len
        end_idx = pad_len + original_len
        acc_processed = acc_filtered[start_idx:end_idx]
        vel_processed = vel[start_idx:end_idx]
        dis_processed = dis[start_idx:end_idx]
    else:
        acc_processed = acc_filtered
        vel_processed = vel
        dis_processed = dis

    return acc_processed, vel_processed, dis_processed
