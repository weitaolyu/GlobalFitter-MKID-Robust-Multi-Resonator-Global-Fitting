# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 22:09:03 2026

@author: NEVER
"""

import os
from typing import Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Import dependencies (ensure these are in your path)
try:
    from main_circle_fit import circle_fit_pratt
    from main_phase_fit import phase_fit
except ImportError:
    print("Warning: circle_fit or phase_fit not found. Ensure they are importable.")

# Constants
PI_2 = 2.0 * np.pi
EPS_TINY = 1e-12


def _estimate_initial_fr_ql(
    f: np.ndarray,
    theta: np.ndarray,
    center: complex
) -> Tuple[float, float, float]:
    """
    Estimate initial guesses of fr, Ql, and theta0 from the phase trajectory.

    Returns
    -------
    fr_guess : float
        Initial guess of resonance frequency.
    Ql_guess : float
        Initial guess of loaded quality factor.
    theta0_guess : float
        Initial guess of resonance phase angle.
    """
    # Resonance point is geometrically opposite to the off-resonant point (1,0)
    target_theta = np.angle(1.0 - center) + np.pi

    # Align target phase to the same unwrap branch as theta
    theta_mean = np.mean(theta)
    n_wraps = np.round((theta_mean - target_theta) / (2 * np.pi))
    target_theta_adjusted = target_theta + n_wraps * 2 * np.pi

    # Closest measured point to the target phase
    idx_closest = np.argmin(np.abs(theta - target_theta_adjusted))

    # Local window for inverse interpolation
    win = min(len(f) // 4, 50)
    idx_min = max(0, idx_closest - win)
    idx_max = min(len(f), idx_closest + win + 1)

    f_local = f[idx_min:idx_max]
    theta_local = theta[idx_min:idx_max]

    # Default fallback
    fr_guess = float(f[idx_closest])
    Ql_guess = 10000.0

    # Try inverse interpolation: theta -> f
    inv_interp = None
    try:
        inv_interp = interp1d(
            theta_local,
            f_local,
            kind='linear',
            assume_sorted=False,
            bounds_error=True
        )
        fr_guess = float(inv_interp(target_theta_adjusted))
    except Exception:
        pass

    # Try geometric bandwidth method using +/- pi/2 phase offsets
    if inv_interp is not None:
        try:
            f_high = float(inv_interp(target_theta_adjusted - np.pi / 2))
            f_low  = float(inv_interp(target_theta_adjusted + np.pi / 2))
            bw_guess = abs(f_high - f_low)
            if bw_guess > EPS_TINY:
                Ql_guess = fr_guess / bw_guess
        except Exception:
            pass

    # Fallback: local slope method
    if not np.isfinite(Ql_guess) or Ql_guess <= 0:
        span = max(3, len(f) // 20)
        s_idx = max(0, idx_closest - span)
        e_idx = min(len(f), idx_closest + span + 1)

        if e_idx - s_idx > 2:
            slope = np.polyfit(f[s_idx:e_idx], theta[s_idx:e_idx], 1)[0]
            Ql_guess = abs(slope) * fr_guess / 2.0
        else:
            Ql_guess = 10000.0

    theta0_guess = float(target_theta_adjusted)
    return fr_guess, Ql_guess, theta0_guess


def _run_phase_fit(
    f: np.ndarray,
    theta: np.ndarray,
    fr_guess: float,
    Ql_guess: float,
    theta0_guess: float,
    use_interp_prefit: bool = True,
    interp_factor: int = 4
) -> Tuple[float, float, float]:
    """
    Run phase fitting in a local window around resonance.

    Strategy
    --------
    1) Use points within +/- 3 linewidths around fr_guess.
    2) Optional: first do a pre-fit on interpolated phase data
       to stabilize the optimizer.
    3) Final fit is always performed on the original phase data.
    """
    # Restrict fit range to +/- 3 linewidths
    f_bw = fr_guess / max(Ql_guess, EPS_TINY)
    valid_mask = (f > fr_guess - 3 * f_bw) & (f < fr_guess + 3 * f_bw)

    if np.sum(valid_mask) > 10:
        f_fit = f[valid_mask]
        theta_fit = theta[valid_mask]
    else:
        f_fit = f
        theta_fit = theta

    # Initial values
    theta0_init, Ql_init, fr_init = theta0_guess, Ql_guess, fr_guess

    # ------------------------------------------------------
    # Optional interpolated pre-fit:
    # This may improve stability, but not add new information.
    # ------------------------------------------------------
    if use_interp_prefit and len(f_fit) >= 4:
        try:
            n_dense = max(len(f_fit) * interp_factor, len(f_fit))
            f_dense = np.linspace(f_fit.min(), f_fit.max(), n_dense)
            theta_dense = np.interp(f_dense, f_fit, theta_fit)

            theta0_init, Ql_init, fr_init = phase_fit(
                f_dense, theta_dense,
                theta0_init, Ql_init, fr_init
            )
        except Exception:
            # Keep original guesses if pre-fit fails
            theta0_init, Ql_init, fr_init = theta0_guess, Ql_guess, fr_guess

    # ------------------------------------------------------
    # Final fit on original phase data
    # ------------------------------------------------------
    try:
        theta0, Ql0, fr0 = phase_fit(
            f_fit, theta_fit,
            theta0_init, Ql_init, fr_init
        )
    except Exception as e:
        print(f"Phase fit failed: {e}")
        theta0, Ql0, fr0 = theta0_init, Ql_init, fr_init

    return float(theta0), float(Ql0), float(fr0)


def _derive_q_parameters(
    Ql: float,
    k: float,
    theta0: float
) -> Tuple[float, float, float]:
    """
    Derive phi, Qc magnitude, and Qi from fitted Ql, k, and theta0.

    Notes
    -----
    This keeps the same logic as the original code:
    - phi0 = theta0 - pi
    - |Qc| = Ql / k
    - Qi uses the magnitude approximation for stability
    """
    phi0 = theta0 - np.pi
    phi0 = (phi0 + np.pi) % (2 * np.pi) - np.pi  # normalize to [-pi, pi]

    Ql = abs(Ql)
    Qc_mag = Ql / max(k, EPS_TINY)

    # Magnitude-based approximation (same logic as original)
    inv_qi = 1.0 / Ql - np.cos(phi0) / Qc_mag

    if inv_qi > 1e-10:
        Qi = 1.0 / inv_qi
    else:
        Qi = 1e9  # high-Q limit

    return float(phi0), float(Qc_mag), float(Qi)


def fit_single_notch_local(
    f: np.ndarray,
    S_flat: np.ndarray,
    plot: bool = False,
    # save_path: Optional[str] = r'C:\Users\NEVER\Downloads\single',
    save_path: Optional[str] = None,
    dpi: int = 900,
    figsize: tuple = (8, 6),
    use_interp_prefit: bool = True,
    interp_factor: int = 4
) -> Dict[str, float]:
    """
    Fit a single notch resonance using Circle Fit + Phase Fit.

    Workflow
    --------
    1. Algebraic circle fit (Pratt)
    2. Extract unwrapped phase on the fitted circle
    3. Estimate initial fr and Ql from geometry + inverse interpolation
    4. Nonlinear phase fit (optional interpolated pre-fit + final raw fit)
    5. Compute derived quantities: phi, Qc, Qi

    Parameters
    ----------
    f : np.ndarray
        Frequency array (assumed monotonic increasing).
    S_flat : np.ndarray
        Complex transmission data after delay/baseline calibration.
    plot : bool
        Whether to show the diagnostic plot.
    save_path : str or None
        Path for saving the plot.
    dpi : int
        Plot resolution.
    figsize : tuple
        Figure size.
    use_interp_prefit : bool
        If True, use interpolated phase for a pre-fit before final fitting
        on the original phase data.
    interp_factor : int
        Multiplicative factor for dense interpolation in the pre-fit stage.

    Returns
    -------
    Dict[str, float]
        Fitted single-resonator parameters:
        fr, Ql, k, phi, Qi, Qc
    """
    # ------------------------------------------------------
    # Step 1: Algebraic Circle Fit (Pratt)
    # ------------------------------------------------------
    xc, yc, r, _ = circle_fit_pratt(S_flat, refine=True)
    center = xc + 1j * yc

    # Initial geometric parameters
    phi_geom = float(np.angle(1.0 - center))
    k0 = float(max(2.0 * r, 1e-8))  # notch diameter approx: k = Ql / |Qc|

    # ------------------------------------------------------
    # Step 2: Extract phase on the fitted circle
    # ------------------------------------------------------
    vec_z = S_flat - center
    theta = np.unwrap(np.angle(vec_z))

    # ------------------------------------------------------
    # Step 3: Estimate fr_guess, Ql_guess, theta0_guess
    # ------------------------------------------------------
    fr_guess, Ql_guess, theta0_guess = _estimate_initial_fr_ql(f, theta, center)

    # ------------------------------------------------------
    # Step 4: Nonlinear phase fit
    # ------------------------------------------------------
    theta0, Ql0, fr0 = _run_phase_fit(
        f=f,
        theta=theta,
        fr_guess=fr_guess,
        Ql_guess=Ql_guess,
        theta0_guess=theta0_guess,
        use_interp_prefit=use_interp_prefit,
        interp_factor=interp_factor
    )

    # ------------------------------------------------------
    # Step 5: Derived parameters
    # ------------------------------------------------------
    phi0, Qc_mag, Qi = _derive_q_parameters(Ql0, k0, theta0)
    # print('Qi', float(Qi) )
    # ------------------------------------------------------
    # Plotting
    # ------------------------------------------------------
    if plot:
        _plot_single_fit(
            f=f,
            S=S_flat,
            xc=xc,
            yc=yc,
            r=r,
            center=center,
            fr=fr0,
            Ql=abs(Ql0),
            Qi=Qi,
            k=k0,
            phi=phi0,
            theta=theta,
            theta0=theta0,
            save_path=save_path,
            dpi=dpi,
            figsize=figsize
        )

    return {
        'fr': float(fr0),
        'Ql': float(abs(Ql0)),
        'k': float(k0),
        'phi': float(phi0),
        'Qi': float(Qi),
        'Qc': float(Qc_mag),
    }


def _plot_single_fit(f, S, xc, yc, r, center, fr, Ql, Qi, k, phi, theta, theta0, save_path, dpi, figsize):
    """Helper function for plotting (2x2 Layout with unified color scheme)"""
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    
    # =========================
    # 统一配色
    # =========================
    data_color = '#0000FF'   # 蓝：原始数据
    fit_color  = '#FFA500'   # 橙：拟合/插值/模型
    mark_color = '#008000'   # 绿：关键标记点
    
    # 辅助颜色（尽量少用，仅给必要 marker 做区分）
    aux_color_1 = '#CC00CC'  # 紫：fr0 marker
    aux_color_2 = '#CC0000'  # 深红：symmetric point / center 等特殊点
    guide_color = '#666666'  # 灰：辅助几何线
    
    # =========================
    # 设置论文作图样式
    # =========================
    plt.rcParams.update({
        'figure.dpi': dpi,
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'figure.titlesize': 14,
        'font.family': 'serif',
        'mathtext.fontset': 'stix'
    })
    
    # ========================================================
    # --- 为画图计算中间几何变量 ---
    # ========================================================
    freq_interp = np.linspace(np.min(f), np.max(f), 100)
    angles_interp = np.interp(freq_interp, f, theta)
    circle_points = center + r * np.exp(1j * angles_interp)
    
    point_1_0 = 1.0 + 0j
    point_sym = center - (point_1_0 - center)
    distances = np.abs(circle_points - point_sym)
    fr0_idx = np.argmin(distances)
    fr_guess = freq_interp[fr0_idx]
    
    # 仅用于显示：频率取整到小数点前
    fr_guess_disp = int(round(fr_guess))
    fr_disp = int(round(fr))

    freq_fine = np.linspace(f.min(), f.max(), 200)
    
    # =========================
    # 创建图像
    # =========================
    fig = plt.figure(figsize=figsize)
    
    # ========================================================
    # (a) Step 1: Circle fitting
    # ========================================================
    ax1 = plt.subplot(2, 2, 1)
    
    # 原始数据：蓝色
    ax1.plot(np.real(S), np.imag(S), '.', color=data_color, alpha=0.65, markersize=5, label='Original data')
    
    # 拟合圆：橙色
    theta_circle = np.linspace(0, 2*np.pi, 200)
    circle_x = xc + r * np.cos(theta_circle)
    circle_y = yc + r * np.sin(theta_circle)
    ax1.plot(circle_x, circle_y, '-', color=fit_color, linewidth=1.8, label='Fitted circle')
    
    # 关键点（marker 允许保留区分）
    ax1.plot(xc, yc, 'o', color=aux_color_2, markersize=6, label=f'Center: ({xc:.3f}, {yc:.3f})')
    ax1.plot(1, 0, '*', color=mark_color, markersize=8, label='Point (1,0)')
    ax1.plot([xc, 1], [yc, 0], '--', color=guide_color, alpha=0.6, linewidth=1.0, label='Center to (1,0)')
    
    ax1.set_xlabel('Re($S_{21}$)')
    ax1.set_ylabel('Im($S_{21}$)')
    ax1.set_title('(a) Step 1: Circle Fitting')
    ax1.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax1.grid(True, alpha=0.3, ls='--', lw=0.6)
    ax1.axis('equal')
    
    textstr = f'Center: ({xc:.3f}, {yc:.3f})\nRadius: {r:.3f}\n$k$: {k:.3f}\n$\\phi$: {phi:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.85, pad=0.5)
    ax1.text(
        0.05, 0.90, textstr, transform=ax1.transAxes, fontsize=8.5,
        verticalalignment='top', bbox=props
    )
    
    # ========================================================
    # (b) Step 2 & 3: Interpolation and frequency remapping
    # ========================================================
    ax2 = plt.subplot(2, 2, 2)
    
    # 频率映射：保留 colormap（例外）
    sc = ax2.scatter(
        np.real(circle_points), np.imag(circle_points),
        c=freq_interp/1e9, cmap='viridis', s=24, alpha=0.9,
        label='Frequency mapping'
    )
    
    # 插值圆：橙色（统一 fit）
    ax2.plot(
        np.real(circle_points), np.imag(circle_points), '-',
        color=fit_color, linewidth=1.8, alpha=0.75, label='Interpolated circle'
    )
    
    # 标记点
    ax2.plot(xc, yc, 'o', color=aux_color_2, markersize=6, label='Center')
    ax2.plot(
        np.real(circle_points[0]), np.imag(circle_points[0]), '^',
        color=mark_color, markersize=7, label='Start ($f_{\\min}$)'
    )
    ax2.plot(
        np.real(circle_points[-1]), np.imag(circle_points[-1]), 'v',
        color=aux_color_1, markersize=7, label='End ($f_{\\max}$)'
    )
    
    cbar = plt.colorbar(sc, ax=ax2, pad=0.05, fraction=0.046)
    cbar.set_label('Frequency (GHz)', fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    
    ax2.set_xlabel('Re($S_{21}$)')
    ax2.set_ylabel('Im($S_{21}$)')
    ax2.set_title('(b) Step 2 & 3: Interpolation and Frequency Remapping')
    ax2.legend(loc='upper left', fontsize=8, framealpha=0.9)
    ax2.grid(True, alpha=0.3, ls='--', lw=0.6)
    ax2.axis('equal')
    
    # ========================================================
    # (c) Step 4: Finding resonant frequency
    # ========================================================
    ax3 = plt.subplot(2, 2, 3)
    
    # 插值圆轨迹：这里建议用橙色，和“拟合/插值”统一
    ax3.plot(
        np.real(circle_points), np.imag(circle_points), '-',
        color=fit_color, alpha=0.8, linewidth=1.6, label='Interpolated circle'
    )
    
    # 关键点
    ax3.plot(1, 0, '*', color=mark_color, markersize=8, label='Point (1,0)')
    ax3.plot(
        np.real(point_sym), np.imag(point_sym), '*',
        color=aux_color_2, markersize=7, label='Symmetric point'
    )
    ax3.plot(
        np.real(circle_points[fr0_idx]), np.imag(circle_points[fr0_idx]),
        'o', color=aux_color_1, markersize=7, label=rf'$f_{{r,0}}$ = {fr_guess_disp}Hz'
    )
    
    # 辅助几何线：灰色
    ax3.plot([1, np.real(point_sym)], [0, np.imag(point_sym)], '--', color=guide_color, alpha=0.6, linewidth=1.0)
    ax3.plot([xc, xc], [yc - 1.2*r, yc + 1.2*r], ':', color=guide_color, alpha=0.4, linewidth=0.8)
    ax3.plot([xc - 1.2*r, xc + 1.2*r], [yc, yc], ':', color=guide_color, alpha=0.4, linewidth=0.8)
    
    ax3.set_xlabel('Re($S_{21}$)')
    ax3.set_ylabel('Im($S_{21}$)')
    ax3.set_title('(c) Step 4: Finding Resonant Frequency')
    ax3.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax3.grid(True, alpha=0.3, ls='--', lw=0.6)
    ax3.axis('equal')
    
    # 不再添加左下角文本框
    
    # ========================================================
    # (d) Step 5: Final phase fit
    # ========================================================
    ax4 = plt.subplot(2, 2, 4)
    
    # 原始相位数据：蓝色
    ax4.plot(f, theta, '.', color=data_color, alpha=0.65, markersize=5, label='Phase data (unwrapped)')
    
    # 最终拟合曲线：橙色
    phase_fit_fine = theta0 + 2 * np.arctan(2 * Ql * (1 - freq_fine / fr))
    ax4.plot(freq_fine, phase_fit_fine, '-', color=fit_color, linewidth=1.8, label='Final phase fit')
    
    # 最终 fr 标记：绿色
    ax4.axvline(
        fr, color=mark_color, linestyle='--', alpha=0.8, linewidth=1.1,
        label=rf'$f_r$ = {fr_disp} Hz'
    )

    ax4.plot(fr, theta0, 'o', color=mark_color, markersize=6)
    
    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('Phase (rad)')
    ax4.set_title('(d) Step 5: Phase Data vs. Final Phase Fit')
    ax4.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax4.grid(True, alpha=0.3, ls='--', lw=0.6)
    
    # final parameters：左下角，小一点
    textstr = (
        f'Final parameters:\n'
        f'$f_r$ = {fr_disp} Hz\n'
        f'$Q_l$ = {Ql:.1f}\n'
        f'$\\phi$: {phi:.3f}\n'
        f'$k$ = {k:.3f}'
    )
    props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.85, pad=0.4)
    ax4.text(
        0.05, 0.08, textstr, transform=ax4.transAxes, fontsize=7.5,
        verticalalignment='bottom', horizontalalignment='left', bbox=props
    )
    
    # =========================
    # 布局调整
    # =========================
    plt.subplots_adjust(
        left=0.09, right=0.97, top=0.93, bottom=0.08,
        wspace=0.32, hspace=0.35
    )
    
    # =========================
    # 保存与显示
    # =========================
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        ext = os.path.splitext(save_path)[1].lower()
        if ext not in ['.png', '.jpg', '.jpeg', '.tiff', '.pdf', '.svg']:
            save_path = save_path + '.png'
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
        print(f"Single fit plot saved to: {save_path}")
    
    plt.show()
