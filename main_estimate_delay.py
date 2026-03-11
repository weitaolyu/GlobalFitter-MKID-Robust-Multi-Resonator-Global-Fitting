# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 17:37:20 2025

@author: NEVER
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import os
from datetime import datetime

# 定义常量
PI_2 = 2 * np.pi

def estimate_tau_robust(f: np.ndarray, S: np.ndarray, plot: bool = False) -> float:
    """
    用 unwrap 相位对频率做鲁棒线性回归，斜率 ≈ -2π τ
    
    参数:
        f: 频率数组 (Hz)
        S: 复数频率响应数组
        plot: 是否绘制拟合图形 (默认False)
    
    返回:
        估计的时间延迟τ (秒)
    """
    # 按频率排序
    idx = np.argsort(f)
    f_sorted = f[idx].astype(float, copy=False)
    S_sorted = S[idx].astype(complex, copy=False)
    
    # 计算解缠相位
    phi = np.unwrap(np.angle(S_sorted))
    
    # 初始最小二乘拟合
    m0, b0 = np.polyfit(f_sorted, phi, 1)
    
    # 鲁棒拟合
    def resid(p):
        m, b = p
        return phi - (m * f_sorted + b)
    
    # 使用鲁棒最小二乘法
    res = least_squares(resid, x0=np.array([m0, b0], float), loss='soft_l1', f_scale=1.0)
    m, b = res.x
    tau = -m / PI_2
    
    plt.rcParams.update({
        'figure.dpi': 600,
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 7,
        'figure.titlesize': 12,
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix'
    })
    
    if plot:
        # Create figure
        fig = plt.figure(figsize=(6, 3))
        
        # Plot original phase data
        plt.scatter(f_sorted, phi, label='Original Phase', color='blue', alpha=0.6, s=5)
        
        # Plot least squares fit line
        plt.plot(f_sorted, m0 * f_sorted + b0, 
                 label=f'Least Squares Fit (τ={-m0/PI_2:.3e}s)', 
                 color='red', linestyle='--', linewidth=2)
        
        # Plot robust fit line
        plt.plot(f_sorted, m * f_sorted + b, 
                 label=f'Robust Fit (τ={tau:.3e}s)', 
                 color='green', linestyle='-', linewidth=1)
        
        plt.xlabel('Frequency (Hz)', fontsize=10)
        plt.ylabel('Unwrapped Phase (rad)', fontsize=10)
        plt.title('Phase-Frequency Relationship and τ Fitting', fontsize=11)
        plt.legend(loc='best', framealpha=0.9, borderpad=0.3, labelspacing=0.2)
        plt.grid(True, ls='--', alpha=0.3, lw=0.5)
        
        # Add tau value annotation
        plt.annotate(f'Estimated τ = {tau:.3e} s', 
                    xy=(0.25, 0.9), xycoords='axes fraction',
                    bbox=dict(boxstyle="round", fc="w", alpha=0.8))
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        save_path = r"C:\Users\NEVER\Downloads"  # Modify this path as needed
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tau_fit_{timestamp}.png"
        full_path = os.path.join(save_path, filename)
        fig.savefig(full_path, dpi=600, bbox_inches='tight', pad_inches=0.1)
        print(f"Figure saved to: {full_path}")
        
        plt.show()
    
    return float(tau)

# 使用示例
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    f = np.linspace(100, 1000, 100)  # 频率范围 100-1000 Hz
    tau_true = 0.001  # 真实延迟 1ms
    
    # 生成带噪声的复数频率响应
    S = np.exp(-1j * 2 * np.pi * f * tau_true)
    noise = 0.1 * (np.random.randn(len(f)) + 1j * np.random.randn(len(f)))
    S_noisy = S + noise
    
    # 调用函数估计τ
    tau_estimated = estimate_tau_robust(f, S_noisy, plot=True)
    
    print(f"真实τ: {tau_true:.6f} s")
    print(f"估计τ: {tau_estimated:.6f} s")
    print(f"相对误差: {abs(tau_estimated - tau_true)/tau_true*100:.2f}%")
