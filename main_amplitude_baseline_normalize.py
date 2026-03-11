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
        # Set Plot Style
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)
        try:
            plt.rcParams['font.family'] = 'serif'
            plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']
        except: pass
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['axes.unicode_minus'] = False
        
        # Create figure
        fig = plt.figure(figsize=(7, 10), dpi=150)
        gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 1.2, 1.5])
        
        # 1. Amplitude
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(frequency_sorted / 1e9, magnitude_db, 'b-', lw=1.5, alpha=0.6, label='Raw')
        ax1.plot(frequency_sorted / 1e9, baseline_db, 'orange', lw=1.5, ls='--', label='Baseline (ASLS)')
        ax1.plot(frequency_sorted / 1e9, 20*np.log10(np.abs(signal_normalized) + EPS_DEFAULT), 
                'g-', lw=1.0, label='Corrected')
        ax1.set_ylabel('Magnitude (dB)'); ax1.legend(loc='lower right', fontsize=8)
        ax1.set_title('(a) Amplitude Baseline Removal')
        ax1.grid(True, alpha=0.3)
        
        # 2. Phase
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(frequency_sorted / 1e9, phase_unwrapped, 'b-', lw=1.5, alpha=0.6, label='Raw Unwrapped')
        ax2.plot(frequency_sorted / 1e9, baseline_phase, 'orange', lw=1.5, ls='--', label='Baseline (ASLS)')
        ax2.plot(frequency_sorted / 1e9, phase_unwrapped_after, 'g-', lw=1.0, label='Corrected')
        ax2.set_ylabel('Phase (rad)'); ax2.legend(loc='lower right', fontsize=8)
        ax2.set_title('(b) Phase Baseline Removal')
        ax2.grid(True, alpha=0.3)
        
        # 3. IQ Plane
        ax3 = fig.add_subplot(gs[2])
        # Downsample for IQ plot to avoid clutter
        step = max(1, len(signal_sorted)//2000)
        ax3.plot(signal_normalized.real[::step], signal_normalized.imag[::step], 
                'm.', ms=2, alpha=0.5, label='Corrected Data')
        # Add unit circle
        theta = np.linspace(0, 2*np.pi, 100)
        ax3.plot(np.cos(theta), np.sin(theta), 'k--', lw=0.8, alpha=0.5)
        
        ax3.axis('equal')
        ax3.set_xlabel('I'); ax3.set_ylabel('Q')
        ax3.set_title('(c) IQ Plane (After Correction)')
        ax3.legend(loc='upper right', fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        save_dir = os.path.join(os.path.expanduser("~"), "Downloads")
        if os.path.exists(save_dir):
            t_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            f_path = os.path.join(save_dir, f"baseline_check_{t_str}.png")
            try:
                fig.savefig(f_path, dpi=300)
                print(f"Baseline plot saved to: {f_path}")
            except: pass
            
        plt.show()

    return frequency_sorted, signal_normalized, baseline_amplitude, baseline_db, baseline_phase