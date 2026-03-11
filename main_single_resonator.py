# -*- coding: utf-8 -*-
"""
Created on Tue Dec 23 10:47:12 2025
Updated on Mon Mar 03 2026

@author: NEVER
"""

import numpy as np
from typing import Dict, Optional
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os

# Import dependencies (ensure these are in your path)
try:
    from main_circle_fit import circle_fit_pratt
    from main_phase_fit import phase_fit
except ImportError:
    print("Warning: circle_fit or phase_fit not found. Ensure they are importable.")

# Constants
PI_2 = 2.0 * np.pi
EPS_TINY = 1e-12

def fit_single_notch_local(
    f: np.ndarray, 
    S_flat: np.ndarray, 
    plot: bool = False,
    save_path: Optional[str] = None, #r'C:\Users\NEVER\Downloads\single'
    dpi: int = 600,
    figsize: tuple = (8, 6) # Slightly larger default
) -> Dict[str, float]:
    """
    Fit a single notch resonance using Circle Fit + Phase Fit method.
    Robust against noise and phase offsets.
    """
    
    # --- Step 1: Algebraic Circle Fit (Pratt's Method) ---
    # This finds the best fit circle in the complex plane
    xc, yc, r, _ = circle_fit_pratt(S_flat, refine=True)
    center = xc + 1j * yc
    
    # Initial guesses from circle geometry
    # Off-resonance point is approx 1+0j (after normalization)
    # The angle of (1 - center) gives the rotation phi
    phi0 = float(np.angle(1.0 - center))
    
    # Diameter = k_c / Q_l ? No, |S_min| = 1 - Ql/Qc_real
    # For a notch, Diameter D = Ql / |Qc| (approx)
    # k0 here is |1/Qc_norm| approx = 2*r
    k0 = float(max(2.0 * r, 1e-8))
    
    # --- Step 2: Phase Extraction ---
    # We remove the electrical delay before this function, 
    # so we focus on the resonance phase circle.
    
    # Vector from center to data points
    vec_z = S_flat - center
    # Unwrap phase angle of these vectors
    theta = np.unwrap(np.angle(vec_z))
    
    # --- Step 3: 高精度寻找 fr 与 Ql 初值 (增强版) ---
    target_theta = np.angle(1.0 - center) + np.pi
    
    # 调整 theta 使其连续，并与 target_theta 处于同一相位周期
    theta_mean = np.mean(theta)
    n_wraps = np.round((theta_mean - target_theta) / (2*np.pi))
    target_theta_adjusted = target_theta + n_wraps * 2*np.pi
    
    # 为了避免相位跳变引起的非单调问题，我们只取中心附近单调的一段
    idx_closest = np.argmin(np.abs(theta - target_theta_adjusted))
    
    # 1. [改进] 亚网格级 (Sub-grid) 插值寻找 fr_guess
    # 在谐振点附近取一个局部窗口做单调插值
    win = min(len(f)//4, 50)
    idx_min = max(0, idx_closest - win)
    idx_max = min(len(f)-1, idx_closest + win)
    
    # 确保用于插值的频率和相位是严格单调的
    f_local = f[idx_min:idx_max]
    theta_local = theta[idx_min:idx_max]
    
    try:
        # 反向插值：通过相位找频率
        inv_interp = interp1d(theta_local, f_local, kind='linear', assume_sorted=False)
        fr_guess = float(inv_interp(target_theta_adjusted))
    except:
        fr_guess = f[idx_closest] # 退回原始方法
        
    # 2. [改进] 使用 3dB 几何带宽法取代求导法估算 Ql_guess
    # 寻找相位偏离中心 +/- 90度 (pi/2) 的频率点
    try:
        f_3db_high = float(inv_interp(target_theta_adjusted - np.pi/2))
        f_3db_low  = float(inv_interp(target_theta_adjusted + np.pi/2))
        bw_guess = abs(f_3db_high - f_3db_low)
        Ql_guess = fr_guess / bw_guess
    except:
        # 如果超出插值范围（可能Q极高或截断），退回原来的导数法
        span = max(3, len(f)//20)
        s_idx, e_idx = max(0, idx_closest - span), min(len(f), idx_closest + span)
        if e_idx - s_idx > 2:
            slope = np.polyfit(f[s_idx:e_idx], theta[s_idx:e_idx], 1)[0]
            Ql_guess = np.abs(slope) * fr_guess / 2.0
        else:
            Ql_guess = 10000.0

    theta0_guess = target_theta_adjusted

    # --- Step 4: Non-linear Phase Fit (加权增强版) ---
    # Fits theta(f) = theta0 + 2*arctan(2*Ql*(f-fr)/fr)
    try:
        # [改进] 强烈建议在你的 phase_fit 函数内部加入权重
        # 权重公式可以是: weights = 1.0 / (1.0 + (2 * Ql_guess * (f - fr_guess) / fr_guess)**2)
        # 如果你无法修改外部的 phase_fit，可以通过只截取中心附近的数据来进行拟合：
        
        # 截取 +/- 3 倍线宽内的数据点给非线性拟合器（过滤掉远端堆积的废点）
        f_bw = fr_guess / Ql_guess
        valid_mask = (f > fr_guess - 3*f_bw) & (f < fr_guess + 3*f_bw)
        
        # 保证至少有10个点用于拟合
        if np.sum(valid_mask) > 10:
            theta0, Ql0, fr0 = phase_fit(f[valid_mask], theta[valid_mask], theta0_guess, Ql_guess, fr_guess)
        else:
            theta0, Ql0, fr0 = phase_fit(f, theta, theta0_guess, Ql_guess, fr_guess)
            
    except Exception as e:
        print(f"Phase fit failed: {e}")
        theta0, Ql0, fr0 = theta0_guess, Ql_guess, fr_guess
    
    # # --- Step 3: Find Resonance Frequency (fr) ---
    # # Resonance occurs where the phase velocity is maximum (steepest slope)
    # # Or geometrically, closest point to the center-symmetry axis
    
    # # Fit a smoothing spline or polynomial to find derivative max
    # # Simple method: find index closest to the "resonance phase"
    # # The resonance phase on the circle is opposite to the off-resonance point
    # # Off-resonance phase ~ angle(1 - center)
    # # Resonance phase ~ angle(1 - center) + pi
    # target_theta = np.angle(1.0 - center) + np.pi
    
    # # Adjust theta to be continuous around target
    # theta_mean = np.mean(theta)
    # n_wraps = np.round((theta_mean - target_theta) / (2*np.pi))
    # target_theta_adjusted = target_theta + n_wraps * 2*np.pi
    
    # # Find index where theta is closest to target
    # idx_res = np.argmin(np.abs(theta - target_theta_adjusted))
    # fr_guess = f[idx_res]
    
    # # Estimate Ql from phase slope: d(theta)/df = 2Ql/fr
    # # Linear fit around resonance
    # span = max(3, len(f)//20)
    # s_idx = max(0, idx_res - span)
    # e_idx = min(len(f), idx_res + span)
    # if e_idx - s_idx > 2:
    #     slope = np.polyfit(f[s_idx:e_idx], theta[s_idx:e_idx], 1)[0]
    #     Ql_guess = np.abs(slope) * fr_guess / 2.0
    # else:
    #     Ql_guess = 10000.0 # Fallback
        
    # theta0_guess = target_theta_adjusted

    # # --- Step 4: Non-linear Phase Fit ---
    # # Fits theta(f) = theta0 + 2*arctan(2*Ql*(f-fr)/fr)
    # try:
    #     theta0, Ql0, fr0 = phase_fit(f, theta, theta0_guess, Ql_guess, fr_guess)
    # except:
    #     # Fallback if phase fit fails
    #     theta0, Ql0, fr0 = theta0_guess, Ql_guess, fr_guess

    # --- Step 5: Calculate Derived Parameters ---
    # Recalculate phi based on the fitted theta0
    # On the circle, theta0 corresponds to f=fr
    # The vector at resonance is r * exp(j*theta0)
    # The off-resonance point (f->inf) is roughly center + r * exp(j*(theta0 - pi))
    # We defined phi as the rotation of the circle.
    # Standard model: S = 1 - ... exp(j*phi)
    # This implies the circle is rotated by phi.
    phi0 = theta0 - np.pi # Approximate relation
    
    # Normalize phi to [-pi, pi]
    phi0 = (phi0 + np.pi) % (2 * np.pi) - np.pi
    
    Ql = float(np.abs(Ql0))
    # Calculate Qc and Qi
    # |Qc| = Ql / k0
    Qc_mag = Ql / k0
    
    # Complex Qc includes the rotation phi
    Qc = Qc_mag * np.exp(-1j * phi0)
    Qc_real = np.real(Qc)
    
    # Qi = 1 / (1/Ql - 1/Qc_real)
    inv_qi = 1.0/Ql - 1.0/Qc_mag # Using magnitude approximation for stability
    # Or strictly: 1.0/Ql - np.real(1.0/Qc)
    
    if inv_qi > 1e-10:
        Qi = 1.0 / inv_qi
    else:
        Qi = 1e9 # High Q limit

    # --- Plotting ---
    if plot:
        _plot_single_fit(f, S_flat, xc, yc, r, center, fr0, Ql, Qi, k0, phi0, theta, theta0, save_path, dpi, figsize)

    return {
        'fr': float(fr0), 
        'Ql': float(Ql), 
        'k': float(k0), 
        'phi': float(phi0), 
        'Qi': float(Qi),
        'Qc': float(Qc_mag) # Return magnitude for simplicity
    }

def _plot_single_fit(f, S, xc, yc, r, center, fr, Ql, Qi, k, phi, theta, theta0, save_path, dpi, figsize):
    """Helper function for plotting (Updated to 2x2 Layout with Interpolation & Sym point)"""
    import os
    
    # 设置IEEE样式参数，稍微增大字体
    plt.rcParams.update({
        'figure.dpi': dpi,
        'font.size': 9,        
        'axes.titlesize': 10,  
        'axes.labelsize': 9,   
        'xtick.labelsize': 8,  
        'ytick.labelsize': 8,  
        'legend.fontsize': 7,  
        'figure.titlesize': 11, 
        'font.family': 'serif',
        'mathtext.fontset': 'stix'
    })
    
    # ========================================================
    # --- 为画图计算中间几何变量 (已修复插值单调性 Bug) ---
    # ========================================================
    # 1. 频率 f 是严格单调递增的，以此为基准生成 100 个均匀的插值频率
    freq_interp = np.linspace(np.min(f), np.max(f), 100)
    
    # 2. 通过频率插值出对应的相位角度 (安全且准确)
    angles_interp = np.interp(freq_interp, f, theta)
    
    # 3. 计算在拟合圆上的对应坐标点
    circle_points = center + r * np.exp(1j * angles_interp)
    
    # 4. 寻找对称点以及对应的 fr_guess
    point_1_0 = 1.0 + 0j
    point_sym = center - (point_1_0 - center)
    distances = np.abs(circle_points - point_sym)
    fr0_idx = np.argmin(distances)
    fr_guess = freq_interp[fr0_idx]
    
    freq_fine = np.linspace(f.min(), f.max(), 200)
    # ========================================================
    
    fig = plt.figure(figsize=figsize)
    
    # === Step 1: Circle fitting (左上) ===
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(np.real(S), np.imag(S), 'b.', alpha=0.6, markersize=4, label='Original data')
    
    # Plot fitted circle
    theta_circle = np.linspace(0, 2*np.pi, 200)
    circle_x = xc + r * np.cos(theta_circle)
    circle_y = yc + r * np.sin(theta_circle)
    ax1.plot(circle_x, circle_y, 'r-', linewidth=1.5, label='Fitted circle')
    
    # Mark center and point (1,0)
    ax1.plot(xc, yc, 'ro', markersize=5, label=f'Center: ({xc:.3f}, {yc:.3f})')
    ax1.plot(1, 0, 'g*', markersize=6, label='Point (1,0)')
    ax1.plot([xc, 1], [yc, 0], 'k--', alpha=0.5, linewidth=0.8, label='Center to (1,0)')
    
    ax1.set_xlabel('Re(S21)')
    ax1.set_ylabel('Im(S21)')
    ax1.set_title('Step 1: Circle Fitting')
    ax1.legend(loc='upper right', fontsize=6, framealpha=0.9)
    ax1.grid(True, alpha=0.3, ls='--', lw=0.5)
    ax1.axis('equal')
    
    # Add text box with circle parameters
    textstr = f'Center: ({xc:.3f}, {yc:.3f})\nRadius: {r:.3f}\nk: {k:.3f}\nϕ: {phi:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8, pad=0.5)
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=7,
            verticalalignment='top', bbox=props)
            
    # === Step 2 & 3: Interpolated circle with frequency mapping (右上) ===
    ax2 = plt.subplot(2, 2, 2)
    sc = ax2.scatter(np.real(circle_points), np.imag(circle_points), 
                    c=freq_interp, cmap='viridis', s=15, alpha=0.8, 
                    label='Frequency mapping')
    ax2.plot(np.real(circle_points), np.imag(circle_points), 'r-', linewidth=1.5, 
            alpha=0.5, label='Interpolated circle')
    
    ax2.plot(xc, yc, 'ro', markersize=5, label='Center')
    ax2.plot(np.real(circle_points[0]), np.imag(circle_points[0]), 'g^', 
            markersize=6, label='Start (f_min)')
    ax2.plot(np.real(circle_points[-1]), np.imag(circle_points[-1]), 'mv', 
            markersize=6, label='End (f_max)')
    
    # Add colorbar
    cbar = plt.colorbar(sc, ax=ax2, pad=0.05, fraction=0.046)
    cbar.set_label('Frequency (Hz)', fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    
    ax2.set_xlabel('Re(S21)')
    ax2.set_ylabel('Im(S21)')
    ax2.set_title('Step 2 & 3: Interpolation and Frequency Mapping')
    ax2.legend(loc='upper left', fontsize=6, framealpha=0.9)
    ax2.grid(True, alpha=0.3, ls='--', lw=0.5)
    ax2.axis('equal')
    
    # === Step 4: Finding resonant frequency (左下) ===
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(np.real(circle_points), np.imag(circle_points), 'b-', alpha=0.6, 
            label='Interpolated circle')
    ax3.plot(1, 0, 'g*', markersize=6, label='Point (1,0)')
    ax3.plot(np.real(point_sym), np.imag(point_sym), 'r*', markersize=5, 
            label='Symmetric point')
    ax3.plot(np.real(circle_points[fr0_idx]), np.imag(circle_points[fr0_idx]), 
            'mo', markersize=6, label=f'fr_guess = {fr_guess:.6f}')
    
    # Draw symmetry line
    ax3.plot([1, np.real(point_sym)], [0, np.imag(point_sym)], 'k--', alpha=0.5, linewidth=0.8)
    ax3.plot([xc, xc], [yc-1.2*r, yc+1.2*r], 'k:', alpha=0.3, linewidth=0.6)
    ax3.plot([xc-1.2*r, xc+1.2*r], [yc, yc], 'k:', alpha=0.3, linewidth=0.6)
    
    ax3.set_xlabel('Re(S21)')
    ax3.set_ylabel('Im(S21)')
    ax3.set_title('Step 4: Finding Resonant Frequency')
    ax3.legend(loc='upper right', fontsize=6, framealpha=0.9)
    ax3.grid(True, alpha=0.3, ls='--', lw=0.5)
    ax3.axis('equal')
    
    textstr = f'fr_guess = {fr_guess:.6f}\nDistance to sym: {distances[fr0_idx]:.3f}'
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8, pad=0.5)
    ax3.text(0.05, 0.05, textstr, transform=ax3.transAxes, fontsize=7,
            verticalalignment='bottom', bbox=props)
            
    # === Step 5: Phase Data vs. Final Phase Fit (右下) ===
    ax4 = plt.subplot(2, 2, 4)
    # 绘制原始相位数据 (直接使用传入的 theta，它已经被 unwrap 过)
    ax4.plot(f, theta, 'b.', alpha=0.6, markersize=4, label='Phase data (unwrapped)')
    
    # 绘制最终相位拟合曲线
    phase_fit_fine = theta0 + 2 * np.arctan(2 * Ql * (1 - freq_fine / fr))
    ax4.plot(freq_fine, phase_fit_fine, 'r-', linewidth=1.5, label='Final phase fit')
    
    # 标记最终谐振频率
    ax4.axvline(fr, color='g', linestyle='--', alpha=0.7, linewidth=0.8,
               label=f'fr_final = {fr:.6f} Hz')
    ax4.plot(fr, theta0, 'go', markersize=5)
    
    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('Phase (rad)')
    ax4.set_title('Step 5: Phase Data vs. Final Phase Fit')
    ax4.legend(loc='upper right', fontsize=6, framealpha=0.9)
    ax4.grid(True, alpha=0.3, ls='--', lw=0.5)
    
    textstr = f'Final parameters:\nfr = {fr:.6f} Hz\nQl = {Ql:.1f}\nθ0 = {theta0:.3f} rad\nQi = {Qi:.1f}\nk = {k:.3f}'
    props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.8, pad=0.5)
    ax4.text(0.05, 0.95, textstr, transform=ax4.transAxes, fontsize=7,
            verticalalignment='top', bbox=props)
            
    # 调整子图间距
    plt.subplots_adjust(left=0.10, right=0.95, top=0.92, bottom=0.08, wspace=0.35, hspace=0.35)
    
    # 保存与显示
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        ext = os.path.splitext(save_path)[1].lower()
        if ext not in ['.png', '.jpg', '.jpeg', '.tiff', '.pdf', '.svg']:
            save_path = save_path + '.png'
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
        print(f"Single fit plot saved to: {save_path}")
        
    plt.show()