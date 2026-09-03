# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 13:48:48 2026

@author: NEVER
"""

# -*- coding: utf-8 -*-
"""
Multi-resonator S21 simulation and paper-quality visualization

Author: NEVER
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from scipy.optimize import curve_fit
import os


# ============================================================
# Multi-resonator model
# ============================================================
class MultiResonatorModel:
    def __init__(self):
        self.parameters = {}

    def set_parameters(self, tau: float, A_func, alpha_func, resonators: List[Dict]):
        self.parameters = {
            'tau': tau,
            'A_func': A_func,
            'alpha_func': alpha_func,
            'resonators': resonators
        }

    def calculate_S21(self, f: np.ndarray) -> np.ndarray:
        tau = self.parameters['tau']
        A_func = self.parameters['A_func']
        alpha_func = self.parameters['alpha_func']
        resonators = self.parameters['resonators']

        cable_delay = np.exp(-2j * np.pi * f * tau)
        baseline = A_func(f) * np.exp(1j * alpha_func(f))

        resonator_product = np.ones_like(f, dtype=complex)

        for r in resonators:
            f_r, Q_c, Q_i, phi = r['f_r'], r['Q_c'], r['Q_i'], r['phi']
            Q_l = 1.0 / (1.0 / Q_i + np.cos(phi) / Q_c)
            x = f / f_r - 1.0
            S21_i = 1.0 - (Q_l / Q_c) * np.exp(1j * phi) / (1.0 + 2j * Q_l * x)
            resonator_product *= S21_i

        return cable_delay * baseline * resonator_product

def add_complex_noise_snr(S21, snr_db=40, seed=1234):
    """
    给复数 S21 加复高斯白噪声，按目标 SNR(dB) 控制噪声强度

    Parameters
    ----------
    S21 : np.ndarray
        原始复数 S21 数据
    snr_db : float
        目标信噪比（功率定义，单位 dB）
        例如：
            60 -> 很干净
            40 -> 轻微噪声（推荐）
            30 -> 明显噪声
            20 -> 较强噪声
    seed : int
        随机种子

    Returns
    -------
    S21_noisy : np.ndarray
        加噪后的复数 S21
    """
    import numpy as np

    rng = np.random.default_rng(seed)

    # 信号平均功率
    signal_power = np.mean(np.abs(S21)**2)

    # 目标噪声平均功率
    noise_power = signal_power / (10**(snr_db / 10))

    # 复高斯噪声：noise = nr + 1j*ni
    # 若 nr, ni ~ N(0, sigma^2)，则 E[|noise|^2] = 2*sigma^2
    sigma = np.sqrt(noise_power / 2)

    noise_real = rng.normal(0, sigma, size=S21.shape)
    noise_imag = rng.normal(0, sigma, size=S21.shape)

    complex_noise = noise_real + 1j * noise_imag
    S21_noisy = S21 + complex_noise

    return S21_noisy


# ============================================================
# Figure 1: magnitude + detrended phase
# ============================================================
def plot_S21(f, S21, resonators):
    plt.rcParams.update({
        'figure.dpi': 600,
        'savefig.dpi': 600,
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8,
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix',
        'lines.linewidth': 1.3,
    })


    
    data_color = '#0000FF'
    mark_color = '#008000'

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 4))

    # Magnitude
    mag_db = 20 * np.log10(np.abs(S21))
    ax1.plot(f / 1e9, mag_db,lw=2.0, color=data_color)
    ax1.set_ylabel(r'$|S_{21}|$ (dB)')
    ax1.set_title('(a) Amplitude of synthetic data')
    ax1.grid(alpha=0.3)

    for i, r in enumerate(resonators):
        ax1.axvline(r['f_r'] / 1e9, color=mark_color,
                    linestyle='--', alpha=0.4,
                    label=rf'$f_{{r,{i+1}}}$')
    ax1.legend()

    # Phase (unwrap + detrend)
    phase = np.unwrap(np.angle(S21))
    p = np.polyfit(f, phase, 1)
    phase_detrended = phase - np.polyval(p, f)

    ax2.plot(f / 1e9, np.rad2deg(phase_detrended),lw=2.0, color=data_color)
    for r in resonators:
        ax2.axvline(r['f_r'] / 1e9, color=mark_color,
                    linestyle='--', alpha=0.4)

    ax2.set_xlabel('Frequency (GHz)')
    ax2.set_ylabel('Phase (deg)')
    ax2.set_title('(b) Detrended unwrapped phase of synthetic data')
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "simulation.png")
    
    plt.savefig(save_path, facecolor='white', dpi=600)
    print("Saved to:", save_path)

    
    plt.show()


# ============================================================
# Figure 2: delay extraction + complex plane
# ============================================================
def plot_delay_and_complex(f, S21, resonators):
    data_color = '#0000FF'
    fit_color = '#FFA500'
    mark_color = '#008000'
    
        # c_raw = "#0000FF"        # blue
        # c_baseline = "#FFA500"   # orange
        # c_corr = "#008000"       # green

    plt.rcParams.update({
        'font.size': 11,              # ← 原来 9
        'axes.titlesize': 12,          # ← 原来 10
        'axes.labelsize': 11,          # ← 原来 9
        'xtick.labelsize': 10,          # ← 原来 8
        'ytick.labelsize': 10,          # ← 原来 8
        'legend.fontsize': 8,          # ← 原来 8
        'lines.linewidth': 1.2,
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix',
    })

    fig = plt.figure(figsize=(6, 3))
    gs = fig.add_gridspec(1, 2, left=0.08, right=0.98,
                          bottom=0.18, top=0.88, wspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax1.set_box_aspect(1)
    ax2.set_box_aspect(1)
    ax2.set_aspect('equal')

    # Phase + delay fit
    phase = np.unwrap(np.angle(S21))

    def linear(x, a, b):
        return a * x + b

    popt, _ = curve_fit(linear, f, phase)
    slope, intercept = popt
    tau_fit = -slope / (2 * np.pi)

    ax1.plot(f / 1e9, phase,lw=2.0, color=data_color, label='Unwrapped phase')
    ax1.plot(f / 1e9, linear(f, slope, intercept),lw=2.0,
             color=fit_color, linestyle='--',
             label=rf'Fit ($\tau={tau_fit:.2e}$ s)')

    for i, r in enumerate(resonators):
        fr = r['f_r'] / 1e9
        idx = np.argmin(np.abs(f / 1e9 - fr))
        ax1.axvline(fr, color=mark_color, linestyle=':', alpha=0.6)
        ax1.text(fr, phase[idx], rf'$f_{{{i+1}}}$',
                 fontsize=9, color=mark_color,
                 ha='center', va='bottom')

    ax1.set_xlabel('Frequency (GHz)')
    ax1.set_ylabel('Phase (rad)')
    ax1.set_title('(a) ')
    ax1.legend(framealpha=0.85)
    ax1.grid(alpha=0.3)

    # Complex plane
    S21_corr = S21 * np.exp(2j * np.pi * f * tau_fit)

    ax2.plot(S21.real, S21.imag,lw=2.0, color=data_color,
             alpha=0.8, label='Original')
    ax2.plot(S21_corr.real, S21_corr.imag, lw=2.0,color=fit_color,
             alpha=0.9, label='Corrected')

    ax2.set_xlabel(r'Re$(S_{21})$')
    ax2.set_ylabel(r'Im$(S_{21})$')
    ax2.set_title('(b) Complex plane')
    ax2.legend(framealpha=0.9)
    ax2.grid(alpha=0.3)

    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "delay.png")
    
    plt.savefig(save_path, facecolor='white', dpi=600)
    print("Saved to:", save_path)

    plt.show()


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':

    # Baseline functions
    def A_func(f):
        fghz = f / 1e9
        return np.clip(
            0.8 + 0.15 * np.sin(2*np.pi*4*(fghz-4.8))
            - 0.7/(1+(2*(f-5.05e9)/100e6)**2),
            1e-6, 1.0
        )

    def alpha_func(f):
        fghz = f / 1e9
        return (
            0.6 * np.sin(2*np.pi*4*(fghz-4.8)) +
            2.0*(2*(f-5.05e9)/100e6)/(1+(2*(f-5.05e9)/100e6)**2)
        )

    resonators = [
        dict(f_r=5.0e9, Q_c=1e4, Q_i=1e5, phi=0.01),
        dict(f_r=5.1e9, Q_c=1e4, Q_i=1e5, phi=0.02),
        dict(f_r=5.2e9, Q_c=1e4, Q_i=1e5, phi=0.03),
        dict(f_r=5.3e9, Q_c=1e4, Q_i=1e5, phi=0.04),
        dict(f_r=5.4e9, Q_c=1e4, Q_i=1e5, phi=0.30),
    ]

    model = MultiResonatorModel()
    model.set_parameters(
        tau=30e-9,
        A_func=A_func,
        alpha_func=alpha_func,
        resonators=resonators
    )

    f = np.linspace(4.6e9, 5.8e9, 10000)
    # f = np.linspace(4.9e9, 5.5e9, 10000)
    S21 = model.calculate_S21(f)
    
    # 加一点噪声
    S21_noisy = add_complex_noise_snr(S21, snr_db=100, seed=1234)
    
    plot_S21(f, S21_noisy, resonators)
    plot_delay_and_complex(f, S21_noisy, resonators)
    
    file_path = os.path.join(os.getcwd(), 'S21_simulation.txt')

    # 如果你想保存加噪数据，就把 S21 改成 S21_noisy
    S21_to_save = S21_noisy   # 或者 S21
    
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write('Frequency(Hz)\tS21_complex\n')
        for freq, s21_val in zip(f, S21_to_save):
            complex_str = f"{s21_val.real:.12e}{s21_val.imag:+.12e}j"
            file.write(f'{freq:.12e}\t{complex_str}\n')
    
    print(f"\n数据已保存到: {file_path}")