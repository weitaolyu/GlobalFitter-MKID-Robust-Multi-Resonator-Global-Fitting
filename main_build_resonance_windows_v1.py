
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 11:44:59 2025
Updated on Mon Mar 03 2026: Adaptive width and overlap-resolved windowing.

@author: NEVER
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from scipy.signal import find_peaks, peak_widths
import os
from datetime import datetime
import matplotlib


def build_resonance_windows(
    f: np.ndarray,
    S: np.ndarray,
    prominence_threshold: float = 1.5,
    height_threshold: float = -2,
    distance: int = 20,
    min_Q: float = 5000.0,
    max_Q: float = 1e8,
    linewidth_factor: float = 10.0,       # 窗口宽度 = N * 3dB带宽
    min_window_width_hz: float = 200e3,   # 最小窗口宽度
    overlap_keep_ratio: float = 0.5,      # 重叠区域保留比例（0~1）
    min_shared_overlap_hz: float = 10e3,  # 最小保留共享重叠宽度
    plot: bool = False,
    DB_SCALE: float = 20.0,
    EPS_DEFAULT: float = 1e-20,
    EPS_TINY: float = 1e-10
) -> List[Tuple[float, float]]:
    """
    Detect resonance peaks and construct adaptive fitting windows.

    New logic:
    1. Detect valid resonance candidates.
    2. Build one raw window for each valid resonance.
    3. If two neighboring windows overlap, do NOT merge them into one macro-window.
       Instead, resolve the overlap by introducing a soft boundary around the center
       of the overlap region, while retaining a small controlled shared overlap.

    Parameters
    ----------
    f : np.ndarray
        Frequency array.
    S : np.ndarray
        Complex S21 array.
    prominence_threshold : float
        Prominence threshold used in peak detection on the inverted magnitude.
    height_threshold : float
        Height threshold (in dB sense) used in peak detection.
    distance : int
        Minimum separation between detected peaks in sample points.
    min_Q : float
        Minimum allowed Q for a valid resonance candidate.
    max_Q : float
        Maximum allowed Q for a valid resonance candidate.
    linewidth_factor : float
        Total window width = linewidth_factor * 3dB bandwidth, unless limited by min_window_width_hz.
    min_window_width_hz : float
        Minimum total window width in Hz.
    overlap_keep_ratio : float
        Fraction of the original overlap width retained after overlap resolution.
        Example: 0.2 means keep 20% of the overlap width as a shared region.
    min_shared_overlap_hz : float
        Minimum shared overlap to retain after resolving overlap.
    plot : bool
        If True, plot detected resonances and final resolved windows.
    DB_SCALE : float
        Usually 20.0 for magnitude in dB.
    EPS_DEFAULT : float
        Small value to avoid log(0).
    EPS_TINY : float
        Small value to avoid division by zero.

    Returns
    -------
    windows : List[Tuple[float, float]]
        One resolved fitting window per valid resonance.
        Overlapping windows are NOT merged; instead, their boundaries are adjusted.
    """

    f = np.asarray(f, dtype=float)
    S = np.asarray(S, dtype=complex)

    # =========================================================
    # 1. Preprocessing: magnitude in dB and inverted trace
    # =========================================================
    mag_db = DB_SCALE * np.log10(np.maximum(np.abs(S), EPS_DEFAULT))
    mag_db_inverted = -mag_db  # find minima by detecting peaks on inverted trace

    # =========================================================
    # 2. Peak Detection
    # =========================================================
    peaks_idx, _ = find_peaks(
        mag_db_inverted,
        prominence=prominence_threshold,
        height=-height_threshold,
        distance=distance
    )

    # No peak detected at all
    if len(peaks_idx) == 0:
        print("No peaks detected. Returning full range.")
        return [(float(f[0]), float(f[-1]))]

    # =========================================================
    # 3. Estimate 3dB bandwidth and Q
    # =========================================================
    widths_samp, _, left_ips, right_ips = peak_widths(
        mag_db_inverted, peaks_idx, rel_height=0.5
    )

    left_ips = np.clip(left_ips, 0, len(f) - 1)
    right_ips = np.clip(right_ips, 0, len(f) - 1)

    f_left_ips = np.interp(left_ips, np.arange(len(f)), f)
    f_right_ips = np.interp(right_ips, np.arange(len(f)), f)

    bw_hz_est = f_right_ips - f_left_ips
    f0 = f[peaks_idx]

    # Preliminary Q estimate
    Q_est = f0 / np.maximum(bw_hz_est, EPS_TINY)

    # =========================================================
    # 4. Filter Valid Peaks
    # =========================================================
    valid_mask = (Q_est >= min_Q) & (Q_est <= max_Q)

    valid_peaks_idx = peaks_idx[valid_mask]
    valid_bw = bw_hz_est[valid_mask]
    valid_f0 = f0[valid_mask]

    if len(valid_peaks_idx) == 0:
        print("No valid resonances found after Q filtering. Returning full range.")
        return [(float(f[0]), float(f[-1]))]

    # =========================================================
    # 5. Build one raw window per valid resonance
    # =========================================================
    raw_windows = []
    for i, fp in enumerate(valid_f0):
        bw = valid_bw[i]

        # Total window width = max(min_window_width_hz, linewidth_factor * bw)
        half_width = max(min_window_width_hz / 2.0, linewidth_factor * bw / 2.0)

        left = max(f[0], fp - half_width)
        right = min(f[-1], fp + half_width)

        raw_windows.append([float(left), float(right)])

    # =========================================================
    # 6. Resolve overlaps instead of merging windows
    # =========================================================
    # Sort by resonance center to ensure neighboring relation is correct
    order = np.argsort(valid_f0)
    valid_f0_sorted = valid_f0[order]
    valid_peaks_idx_sorted = valid_peaks_idx[order]
    valid_bw_sorted = valid_bw[order]
    raw_windows_sorted = [raw_windows[i] for i in order]

    resolved_windows = [raw_windows_sorted[0][:]]

    for i in range(1, len(raw_windows_sorted)):
        prev_l, prev_r = resolved_windows[-1]
        curr_l, curr_r = raw_windows_sorted[i]

        f_prev = float(valid_f0_sorted[i - 1])
        f_curr = float(valid_f0_sorted[i])

        # -----------------------------------------------------
        # Case 1: No overlap
        # -----------------------------------------------------
        if curr_l >= prev_r:
            resolved_windows.append([curr_l, curr_r])
            continue

        # -----------------------------------------------------
        # Case 2: Overlap exists
        # Overlap region = [curr_l, prev_r]
        # We do NOT merge.
        # Instead, resolve using overlap center + small retained overlap.
        # -----------------------------------------------------
        overlap_l = curr_l
        overlap_r = prev_r
        overlap_width = max(0.0, overlap_r - overlap_l)

        # Center of overlap region
        overlap_center = 0.5 * (overlap_l + overlap_r)

        # Retained shared overlap width
        keep_overlap = max(min_shared_overlap_hz, overlap_keep_ratio * overlap_width)

        # But cannot exceed original overlap width
        keep_overlap = min(keep_overlap, overlap_width)

        half_keep = 0.5 * keep_overlap

        # New boundaries:
        # Left window ends slightly to the right of overlap center
        # Right window starts slightly to the left of overlap center
        new_prev_r = overlap_center + half_keep
        new_curr_l = overlap_center - half_keep

        # Safety clipping:
        # must remain inside original windows
        new_prev_r = min(new_prev_r, prev_r)
        new_curr_l = max(new_curr_l, curr_l)

        # Safety clipping:
        # do not allow the left window to end before its own center
        # do not allow the right window to start after its own center
        new_prev_r = max(new_prev_r, f_prev)
        new_curr_l = min(new_curr_l, f_curr)

        # In pathological cases, ensure ordering is still valid
        if new_prev_r < prev_l:
            new_prev_r = prev_l
        if new_curr_l > curr_r:
            new_curr_l = curr_r

        resolved_windows[-1] = [float(prev_l), float(new_prev_r)]
        resolved_windows.append([float(new_curr_l), float(curr_r)])

    windows = [(float(l), float(r)) for l, r in resolved_windows]

    # =========================================================
    # 7. Plotting
    # =========================================================
    if plot:
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)
        try:
            plt.rcParams['font.family'] = 'serif'
            plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
        except Exception:
            pass
        plt.rcParams['mathtext.fontset'] = 'stix'

        fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)

        # Raw magnitude
        ax.plot(f / 1e9, mag_db, 'b-', lw=1.0, alpha=0.8, label='S21')

        # Valid peaks
        if len(valid_peaks_idx_sorted) > 0:
            ax.scatter(
                valid_f0_sorted / 1e9,
                mag_db[valid_peaks_idx_sorted],
                c='r',
                marker='v',
                s=30,
                zorder=5,
                label='Valid resonance'
            )

        # Draw resolved windows
        for i, (l, r) in enumerate(windows):
            color = plt.cm.tab20(i % 20)
            ax.axvspan(l / 1e9, r / 1e9, color=color, alpha=0.20)

            ax.text(
                (l + r) / 2 / 1e9,
                np.min(mag_db),
                str(i + 1),
                ha='center',
                va='bottom',
                fontsize=8,
                color='k'
            )

        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Magnitude (dB)')
        ax.set_title(
            f'Detected {len(valid_peaks_idx_sorted)} Valid Resonances -> {len(windows)} Resolved Windows'
        )
        ax.grid(True, alpha=0.3, ls='--')
        ax.legend(loc='upper right', fontsize=8)
        plt.tight_layout()

        # Auto-save
        save_dir = os.getcwd()
        t_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f"resolved_windows_{t_str}.png")
        fig.savefig(save_path)
        print(f"Window plot saved to: {save_path}")

        plt.show()

    print(f"Detected {len(valid_peaks_idx_sorted)} valid peaks, generated {len(windows)} resolved windows.")
    return windows