# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 14:11:16 2025
Updated on Mon Mar 03 2026: Enhanced with padding and robust phase smoothing.

@author: NEVER
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from typing import NamedTuple, Tuple
import os
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime

# Define constants
DB_SCALE = 20.0
EPS_DEFAULT = 1e-12
EPS_TINY = 1e-300

# Define parameters structure
class AslsParams(NamedTuple):
    method: str = 'asls'
    asls_lam: float = 1e5           # 增大平滑度，防止过拟合宽峰
    asls_p_upper: float = 0.001     # 减小不对称因子，贴合上包络
    asls_niter: int = 10
    phase_asls_lam: float = 1e7     # 极大值，强迫相位基线接近线性
    phase_asls_p: float = 0.5
    phase_asls_niter: int = 10

def asls_amplitude(y_db: np.ndarray, lam: float = 1e5, p_upper: float = 0.001, niter: int = 10, pad_len: int = 50) -> np.ndarray:
    """
    [Enhanced] Estimate upper baseline using ASLS with reflect padding.
    """
    N_orig = len(y_db)
    # 1. Padding to handle boundary effects
    pad_width = min(pad_len, N_orig // 2)
    y_padded = np.pad(y_db, (pad_width, pad_width), mode='reflect')
    
    N = len(y_padded)
    # 2. Sparse matrix construction
    D = diags([1, -2, 1], [0, 1, 2], shape=(N - 2, N), format='csc')
    DTD = lam * (D.T @ D)
    w = np.ones(N, dtype=float)
    
    # 3. Iterative solver
    z = np.zeros(N)
    for _ in range(niter):
        W = diags(w, 0, format='csc')
        # Solve (W + DTD)z = Wy
        z = spsolve(W + DTD, w * y_padded)
        r = y_padded - z
        # Asymmetric weighting: small weight for negative residuals (dips)
        w = p_upper * (r < 0).astype(float) + (1.0 - p_upper) * (r >= 0).astype(float)
    
    # 4. Remove padding
    return z[pad_width : pad_width + N_orig]

def asls_phase(y: np.ndarray, lam: float = 1e7, p: float = 0.5, niter: int = 10, pad_len: int = 50) -> np.ndarray:
    """
    [Enhanced] Smooth phase using ASLS with edge padding.
    """
    N_orig = len(y)
    # Phase padding uses 'edge' to maintain continuity trend
    pad_width = min(pad_len, N_orig // 2)
    y_padded = np.pad(y, (pad_width, pad_width), mode='edge')
    
    N = len(y_padded)
    D = diags([1, -2, 1], [0, 1, 2], shape=(N - 2, N), format='csc')
    DTD = lam * (D.T @ D)
    w = np.ones(N, dtype=float)
    
    z = np.zeros(N)
    for _ in range(niter):
        W = diags(w, 0, format='csc')
        z = spsolve(W + DTD, w * y_padded)
        r = y_padded - z
        # Symmetric weighting for phase noise
        w = p * (r > 0).astype(float) + (1.0 - p) * (r <= 0).astype(float)
        
    return z[pad_width : pad_width + N_orig]

def amplitude_baseline_normalize(
    frequency: np.ndarray, 
    signal_complex: np.ndarray, 
    params: AslsParams, 
    plot: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize amplitude and phase baseline using enhanced ASLS.
    Returns:
        frequency_sorted, signal_normalized, baseline_amplitude, baseline_db, baseline_phase
    """
    # Ensure sorted frequency
    idx = np.argsort(frequency)
    frequency_sorted = np.asarray(frequency, float)[idx]
    signal_sorted = np.asarray(signal_complex, complex)[idx]

    # --- Amplitude Correction ---
    magnitude_db = DB_SCALE * np.log10(np.abs(signal_sorted) + EPS_DEFAULT)
    baseline_db = asls_amplitude(
        magnitude_db, 
        lam=params.asls_lam, 
        p_upper=params.asls_p_upper, 
        niter=params.asls_niter
    )
    baseline_amplitude = 10.0 ** (baseline_db / DB_SCALE)
    
    # Normalized magnitude
    signal_mag_norm = np.abs(signal_sorted) / (baseline_amplitude + EPS_TINY)

    # --- Phase Correction ---
    # Unwrap phase of the AMPLITUDE-NORMALIZED signal
    # (This removes amplitude-dependent phase noise if any)
    phase_raw = np.angle(signal_sorted) # Use raw phase, not normalized mag phase
    phase_unwrapped = np.unwrap(phase_raw)
    
    # Smooth phase baseline (representing electrical delay + dispersion)
    baseline_phase = asls_phase(
        phase_unwrapped, 
        lam=params.phase_asls_lam, 
        p=params.phase_asls_p, 
        niter=params.phase_asls_niter
    )
    
    # --- Final Normalization ---
    # S_norm = S_raw / (Amp_base * exp(j * Phase_base))
    baseline_complex = baseline_amplitude * np.exp(1j * baseline_phase)
    signal_normalized = signal_sorted / (baseline_complex + EPS_TINY)
    
    # Calculate phase after correction for plotting
    phase_unwrapped_after = np.unwrap(np.angle(signal_normalized))

    if plot:
        # =============================
        # Global style
        # =============================
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
    
            # font size
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 7,
    
            # line / axis style
            "axes.linewidth": 1,
            "lines.linewidth": 1.5,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.25,
    
            # save
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03
        })
    
        # =============================
        # Colors
        # =============================
        c_raw = "#0000FF"        # blue
        c_baseline = "#FFA500"   # orange
        c_corr = "#008000"       # green
        c_raw_iq = "#0000FF"
        c_corr_iq = "#008000"
        c_circle = "#4D4D4D"
    
        # =============================
        # Data
        # =============================
        x_ghz = frequency_sorted / 1e9
    
        # Panel (a): still in dB
        corrected_db = 20 * np.log10(np.abs(signal_normalized) + EPS_DEFAULT)
        corrected_db_neg = -corrected_db   # for visual separation only
    
        # Panel (c)
        raw_iq = signal_sorted
        corr_iq = signal_normalized
    
        # =============================
        # Figure layout
        # =============================
        fig = plt.figure(figsize=(7.2, 8.4), dpi=600, constrained_layout=True)
        gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 1.15], hspace=0.10)
    
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax3 = fig.add_subplot(gs[2])
    
        # =============================
        # Helper
        # =============================
        def beautify_axis(ax):
            # Keep four-side frame
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.9)
    
            ax.grid(True, which="major", alpha=0.25)
    
            # Remove minor ticks
            ax.minorticks_off()
    
            ax.tick_params(direction="in", which="major", top=True, right=True,
                           length=4, width=0.8)
    
        # =============================
        # (a) Amplitude in dB
        # =============================
        ax1.plot(x_ghz, magnitude_db,
                 color=c_raw, lw=2, alpha=0.80, label=r"Raw $|S_{21}|$")
        ax1.plot(x_ghz, baseline_db,
                 color=c_baseline, lw=2, ls=(0, (5, 2)), label="Baseline (ASLS)")
        ax1.plot(x_ghz, corrected_db_neg,
                 color=c_corr, lw=2.5, label=r"Corrected (shown as $-|S_{21}|$ in dB)")
    
        # zero line for reference
        ax1.axhline(0, color="0.4", lw=0.8, ls="--", alpha=0.6)
    
        ax1.set_ylabel(r"$|S_{21}|$ (dB)")
        ax1.set_title("(a) Amplitude Baseline Removal", pad=6)
        ax1.legend(loc="best", frameon=True, edgecolor="0.85", fancybox=False)
        beautify_axis(ax1)
    
        # =============================
        # (b) Phase
        # =============================
        ax2.plot(x_ghz, phase_unwrapped,
                 color=c_raw, lw=2, alpha=0.80, label="Raw unwrapped")
        ax2.plot(x_ghz, baseline_phase,
                 color=c_baseline, lw=2, ls=(0, (5, 2)), label="Baseline (ASLS)")
        ax2.plot(x_ghz, phase_unwrapped_after,
                 color=c_corr, lw=2.5, label="Corrected")
    
        ax2.set_ylabel("Phase (rad)")
        ax2.set_xlabel("Frequency (GHz)")
        ax2.set_title("(b) Phase Baseline Removal", pad=6)
        ax2.legend(loc="best", frameon=True, edgecolor="0.85", fancybox=False)
        beautify_axis(ax2)
    
        # Hide x tick labels for the first panel
        plt.setp(ax1.get_xticklabels(), visible=False)
    
        # =============================
        # (c) IQ plane: Raw vs Corrected
        # =============================
        step = max(1, len(signal_sorted) // 1800)
    
        ax3.plot(raw_iq.real[::step], raw_iq.imag[::step],
                 linestyle="None",
                 marker="o",
                 ms=3,
                 mec="none",
                 color=c_raw_iq,
                 alpha=0.4,
                 label="Raw")
    
        ax3.plot(corr_iq.real[::step], corr_iq.imag[::step],
                 linestyle="None",
                 marker="*",
                 ms=5,
                 mec="none",
                 color=c_corr_iq,
                 alpha=0.6,
                 label="Corrected")
    
        # Unit circle
        theta = np.linspace(0, 2*np.pi, 400)
        ax3.plot(np.cos(theta), np.sin(theta),
                 color=c_circle, lw=1.0, ls=(0, (4, 2)), alpha=0.75, label="Unit circle")
    
        ax3.set_aspect("equal", adjustable="box")
        ax3.set_xlabel(r"Re($S_{21}$)")
        ax3.set_ylabel(r"Im($S_{21}$)")
        ax3.set_title("(c) IQ Plane: Raw vs Corrected", pad=6)
        ax3.legend(loc="best", frameon=True, edgecolor="0.85", fancybox=False)
        beautify_axis(ax3)
    
        # Symmetric axis limits
        iq_r = np.r_[raw_iq.real[::step], corr_iq.real[::step]]
        iq_i = np.r_[raw_iq.imag[::step], corr_iq.imag[::step]]
        lim = 1.08 * max(np.max(np.abs(iq_r)), np.max(np.abs(iq_i)), 1.0)
        ax3.set_xlim(-lim, lim)
        ax3.set_ylim(-lim, lim)
    
        # =============================
        # Save
        # =============================
        save_dir = os.getcwd()
        t_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        f_path = os.path.join(save_dir, f"baseline_check_{t_str}.png")
        try:
            fig.savefig(f_path, dpi=900)
            print(f"Baseline plot saved to: {f_path}")
        except Exception as e:
            print(f"Failed to save figure: {e}")
    
        plt.show()


    return frequency_sorted, signal_normalized, baseline_amplitude, baseline_db, baseline_phase