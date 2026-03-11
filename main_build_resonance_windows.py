# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 11:44:59 2025
Updated on Mon Mar 03 2026: Adaptive width and window merging.

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
    linewidth_factor: float = 10.0,      # 窗口宽度 = N * 3dB带宽
    min_window_width_hz: float = 200e3,  # 最小窗口宽度
    plot: bool = False,
    DB_SCALE: float = 20.0,
    EPS_DEFAULT: float = 1e-20,
    EPS_TINY: float = 1e-10
) -> List[Tuple[float, float]]:
    """
    Detects resonance peaks and creates adaptive windows.
    Overlapping windows are merged to ensure global context.
    """
    f = np.asarray(f, float)
    S = np.asarray(S, complex)
    
    # 1. Preprocessing
    mag_db = DB_SCALE * np.log10(np.maximum(np.abs(S), EPS_DEFAULT))
    mag_db_inverted = -mag_db # Find peaks in inverted magnitude

    # 2. Peak Detection
    peaks_idx, _ = find_peaks(
        mag_db_inverted,
        prominence=prominence_threshold,
        height=-height_threshold,
        distance=distance
    )

    # 3. Estimate Bandwidth and Q
    # rel_height=0.5 corresponds to 3dB width
    widths_samp, _, left_ips, right_ips = peak_widths(
        mag_db_inverted, peaks_idx, rel_height=0.5
    )

    # Interpolate to find frequency bandwidth
    # Use clip to avoid index out of bounds
    left_ips = np.clip(left_ips, 0, len(f)-1)
    right_ips = np.clip(right_ips, 0, len(f)-1)
    
    f_left_ips = np.interp(left_ips, np.arange(len(f)), f)
    f_right_ips = np.interp(right_ips, np.arange(len(f)), f)
    bw_hz_est = f_right_ips - f_left_ips
    f0 = f[peaks_idx]
    
    # Q Estimation
    Q_est = f0 / np.maximum(bw_hz_est, EPS_TINY)

    # 4. Filter Valid Peaks
    valid_mask = (Q_est >= min_Q) & (Q_est <= max_Q)
    valid_peaks_idx = peaks_idx[valid_mask]
    valid_bw = bw_hz_est[valid_mask]
    valid_f0 = f0[valid_mask]
    
    # 5. Generate Adaptive Windows
    raw_windows = []
    for i, fp in enumerate(valid_f0):
        bw = valid_bw[i]
        
        # Window half width = Factor * Bandwidth / 2
        # But respect minimum width
        half_width = max(min_window_width_hz / 2.0, linewidth_factor * bw / 2.0)
        
        left = max(f[0], fp - half_width)
        right = min(f[-1], fp + half_width)
        raw_windows.append([left, right])

    # 6. Merge Overlapping Windows
    if not raw_windows:
        print("No valid resonances found. Returning full range.")
        return [(float(f[0]), float(f[-1]))]
        
    raw_windows.sort(key=lambda x: x[0])
    
    merged_windows = []
    if raw_windows:
        curr_l, curr_r = raw_windows[0]
        for i in range(1, len(raw_windows)):
            next_l, next_r = raw_windows[i]
            
            # Merge if next window starts before current ends (or very close)
            if next_l <= curr_r:
                curr_r = max(curr_r, next_r)
            else:
                merged_windows.append((curr_l, curr_r))
                curr_l, curr_r = next_l, next_r
        merged_windows.append((curr_l, curr_r))

    windows = [(float(l), float(r)) for l, r in merged_windows]

    # --- Plotting ---
    if plot:
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)
        try:
            plt.rcParams['font.family'] = 'serif'
            plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
        except: pass
        plt.rcParams['mathtext.fontset'] = 'stix'
        
        fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
        ax.plot(f/1e9, mag_db, 'b-', lw=1.0, alpha=0.8, label='S21')
        
        # Mark valid peaks
        if len(valid_peaks_idx) > 0:
            ax.scatter(valid_f0/1e9, mag_db[valid_peaks_idx], 
                      c='r', marker='v', s=30, zorder=5, label='Resonance')
        
        # Draw windows
        for i, (l, r) in enumerate(windows):
            color = plt.cm.tab20(i % 20)
            ax.axvspan(l/1e9, r/1e9, color=color, alpha=0.2)
            # Label window
            ax.text((l+r)/2/1e9, np.min(mag_db), str(i+1), 
                   ha='center', va='bottom', fontsize=8, color='k')

        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Magnitude (dB)')
        ax.set_title(f'Detected {len(valid_peaks_idx)} Peaks -> {len(windows)} Windows')
        ax.grid(True, alpha=0.3, ls='--')
        ax.legend(loc='upper right', fontsize=8)
        plt.tight_layout()
        
        # Auto-save
        save_dir = os.path.join(os.path.expanduser("~"), "Downloads")
        if os.path.exists(save_dir):
            t_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            fig.savefig(os.path.join(save_dir, f"windows_{t_str}.png"))
            print(f"Window plot saved to Downloads.")
            
        plt.show()

    print(f"Detected {len(valid_peaks_idx)} peaks, merged into {len(windows)} windows.")
    return windows