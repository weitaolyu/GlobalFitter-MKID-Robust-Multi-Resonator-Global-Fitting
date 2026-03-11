# -*- coding: utf-8 -*-
"""
MKID/超导谐振器 S21 全局拟合算法 (Ultimate Edition)
包含: 动态基线优化 + 严格共享 Qi + 参数归一化 + cos(phi) 物理修正
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import warnings
import argparse
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, NamedTuple, Tuple, Any

try:
    from typing import TypedDict
except ImportError:
    from typing import Dict as TypedDict

from scipy.optimize import least_squares
from itertools import combinations
from scipy.signal import find_peaks

# =========================================================================================
#                                【 用户可调参数配置区 】
# =========================================================================================

# ----------------- 1. 文件与基础配置 -----------------
DEFAULT_FILE = r'100mkoff.s2p'   

# ----------------- 2. 谐振峰检测 (Peak Detection) -----------------
CFG_PROMINENCE = 1.5       
CFG_HEIGHT = -2.0          
CFG_DISTANCE = 20          
CFG_MIN_Q = 6000.0         

# ----------------- 3. 拟合窗口提取 (Window Building) -----------------
CFG_LW_FACTOR = 10.0       
CFG_MIN_WIN_HZ = 500e3     
CFG_MIN_PTS_WIN = 20       

# ----------------- 4. 优化算法控制 (Optimization Control) -----------------
CFG_USE_WEIGHT = True      
CFG_WEIGHT_SIGMA = 3.0     
# 【全新约束控制】
CFG_QI_BOUND_RATIO = 0.5   # [Qi 允许波动范围]: 0.5 表示全局共享的 Qi 只能在初始中位数的 ±50% 范围内波动。
CFG_QC_BOUND_RATIO = 0.5   # [Qc 允许波动范围]: 0.2 表示限制单坑的深浅只能在初值的 ±20% 内波动。

# ----------------- 5. 降采样策略 (Downsampling) -----------------
CFG_DOWNSAMPLE = True      

# ----------------- 6. 基线校准算法 (ASLS Baseline) -----------------
CFG_ASLS_LAM = 1e4         
CFG_ASLS_P_UP = 0.01       
CFG_PHASE_LAM = 5e5        
CFG_PHASE_P = 0.5          

# ----------------- 7. 可视化与调试 (Plot & Debug) -----------------
CFG_PLOT_FINAL = True      
CFG_PLOT_WINDOWS = False   
CFG_PLOT_TAU = False       
CFG_PLOT_PREVIEW = False   
CFG_VERBOSE = True         

# =========================================================================================

# ==================== 依赖检查与导入 ====================
try:
    from main_data_reader import read_s2p, read_txt, read_data
    from main_estimate_delay import estimate_tau_robust
    from main_amplitude_baseline_normalize import amplitude_baseline_normalize
    from main_build_resonance_windows import build_resonance_windows
    from main_single_resonator import fit_single_notch_local
except ImportError as e:
    print(f"[严重错误] 缺失必要的依赖库: {e}")
    sys.exit(1)

# ==================== 配置与常量 ====================
PI_2 = 2.0 * np.pi

def configure_plotting():
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    font_candidates = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif', 'serif']
    for font in font_candidates:
        try:
            plt.rcParams['font.family'] = 'serif'
            plt.rcParams['font.serif'] = [font]
            break
        except Exception:
            continue
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['figure.dpi'] = 150

configure_plotting()
warnings.filterwarnings('ignore', category=UserWarning)

class AslsParams(NamedTuple):
    method: str = 'asls'
    asls_lam: float = CFG_ASLS_LAM
    asls_p_upper: float = CFG_ASLS_P_UP
    asls_niter: int = 10
    phase_asls_lam: float = CFG_PHASE_LAM
    phase_asls_p: float = CFG_PHASE_P
    phase_asls_niter: int = 10

class FitOutput(TypedDict):
    global_params: Dict[str, Any]
    resonators: List[Dict]
    window_info: List[Dict]
    S21_fit: np.ndarray
    S21_fit_cal: np.ndarray
    optimization_scheme: str
    scheme_index_used: int
    downsample_enabled: bool
    weighting_enabled: bool
    downsample_info: Optional[Dict]

# ==================== 核心物理模型 ====================

def s21_model_total(f: np.ndarray, res_params_flat: np.ndarray, baseline_params: np.ndarray, n_res: int) -> np.ndarray:
    A, alpha, tau = baseline_params
    S_baseline = A * np.exp(1j * (alpha + PI_2 * f * tau))
    
    p = res_params_flat.reshape(n_res, 4)
    fr = p[:, 0]
    Ql = p[:, 1]
    Qc = p[:, 2]
    phi = p[:, 3]
    
    k = np.zeros_like(Ql)
    mask = np.abs(Qc) > 1e-9
    k[mask] = Ql[mask] / Qc[mask]
    
    x = (f[:, None] - fr) / fr 
    denom = 1.0 + 2j * Ql * x
    term = k * np.exp(1j * phi) / denom
    
    S_res = np.prod(1.0 - term, axis=1)
    return S_res * S_baseline

def _extract_resonator_arrays(results: List[Dict]) -> np.ndarray:
    return np.array([[res['fr'], res['Ql'], res['phi'], res['k']] for res in results])

def _calculate_qi_from_params(fr: float, Ql: float, phi: float, k: float) -> Tuple[float, float]:
    """【已修正】使用包含 cos(phi) 的精确公式提取 Qi 和 Qc"""
    if abs(k) > 1e-9:
        Qc = Ql / k
        inv_qi = 1.0/Ql - np.cos(phi)/Qc
        if inv_qi > 1e-10: 
            Qi = 1.0 / inv_qi
        else:
            Qi = 1e9 
    else:
        Qc, Qi = 1e9, Ql
    return Qi, Qc

# ==================== 降采样逻辑 ====================

def _adaptive_downsample_with_windows(
    f: np.ndarray, S: np.ndarray, 
    windows: List[Tuple[float, float]], 
    indep_results: List[Dict], 
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """【极限瘦身版降采样】强制控制送入优化器的数据量"""
    n_points = len(f)
    keep_indices = set()
    
    # 1. 保留坑底数据，但【强制限制最大点数】
    for res in indep_results:
        fr_val = res['fr']
        bw_val = fr_val / res['Ql']
        core_idx = np.where((f >= fr_val - 1.5 * bw_val) & (f <= fr_val + 1.5 * bw_val))[0]
        
        # ==========================================
        # 【提速核心】如果坑底点数超过 40 个，强制等距抽样
        # 这既保住了深坑的轮廓，又杜绝了内存和矩阵爆炸
        # ==========================================
        if len(core_idx) > 40:
            step = len(core_idx) // 40
            core_idx = core_idx[::step]
            
        keep_indices.update(core_idx)
    
    # 2. 窗口内其他区域极度降采样 (最多只留20个点)
    for i, (l, r) in enumerate(windows):
        idx = np.where((f >= l) & (f <= r))[0]
        if len(idx) > 20:
            step = len(idx) // 20
            keep_indices.update(idx[::step])
        else:
            keep_indices.update(idx)
        
    # 3. 全局基线极度稀疏化 (整条几十万点的基线，只抽 100 个点用于定稳！)
    baseline_step = max(1, n_points // 100)
    all_idx = np.arange(0, n_points, baseline_step)
    keep_indices.update(all_idx)
    keep_indices.add(0)
    keep_indices.add(n_points - 1)
    
    keep_indices_sorted = sorted(list(keep_indices))
    keep_indices_arr = np.array(keep_indices_sorted)
    
    if verbose: 
        # 你将看到降采样后的点数从几千甚至上万，暴降到几百个点！
        print(f"  降采样: {n_points} -> {len(keep_indices_sorted)} "
              f"({len(keep_indices_sorted)/n_points*100:.2f}%) (已开启强制数据瘦身)")
    
    return f[keep_indices_arr], S[keep_indices_arr], keep_indices_arr
# ==================== 全局优化器 (Ultimate) ====================

def _optimize_global_robust(
    f: np.ndarray, S_data: np.ndarray, 
    independent_results: List[Dict],
    verbose: bool = True,
    weights: Optional[np.ndarray] = None,
    qi_bound_ratio: float = 0.3,  
    qc_bound_ratio: float = 0.2,  
    **kwargs 
) -> Dict:
    n_res = len(independent_results)
    res_arrays = _extract_resonator_arrays(independent_results) 
    
    qi_vals = []
    for row in res_arrays:
        qi, _ = _calculate_qi_from_params(*row)
        if 0 < qi < 1e9: qi_vals.append(qi)
    qi_shared_init = np.median(qi_vals) if qi_vals else 1e5
    
    p_global = [1.0, 0.0, 0.0, qi_shared_init] 
    
    scales = [1.0, 1.0, 1e-9, qi_shared_init]
    qi_min = max(qi_shared_init * (1.0 - qi_bound_ratio), 1000.0) 
    qi_max = qi_shared_init * (1.0 + qi_bound_ratio)
    
    bounds_low = [0.8, -np.pi, -1e-8, qi_min]     
    bounds_high = [1.2, np.pi, 1e-8, qi_max]       
    
    p_local = []
    for i in range(n_res):
        fr, Ql, phi, k = res_arrays[i]
        _, Qc = _calculate_qi_from_params(fr, Ql, phi, k)
        
        p_local.extend([fr, Qc, phi])
        scales.extend([fr, Qc, max(abs(phi), 1.0)])
        
        bw = fr / Ql
        fr_margin = bw * 0.02           
        phi_margin = 0.1                
        qc_min_bound = Qc * (1.0 - qc_bound_ratio) 
        qc_max_bound = Qc * (1.0 + qc_bound_ratio)
        
        bounds_low.extend([fr - fr_margin, qc_min_bound, phi - phi_margin])
        bounds_high.extend([fr + fr_margin, qc_max_bound, phi + phi_margin])
        
    x0_real = np.array(p_global + p_local)
    bounds_low_real = np.array(bounds_low)
    bounds_high_real = np.array(bounds_high)
    scales_arr = np.array(scales)
    
    x0_norm = x0_real / scales_arr
    bounds_norm = (bounds_low_real / scales_arr, bounds_high_real / scales_arr)
    
    w_sqrt = np.sqrt(weights) if weights is not None else 1.0

    def fun(p_norm):
        p_real = p_norm * scales_arr
        base_p = p_real[:3]      
        qi_shared = p_real[3]    
        local_p = p_real[4:]     
        
        # =======================================================
        # 【极致提速：全向量化矩阵运算，彻底抛弃 for 循环】
        # =======================================================
        fr_arr = local_p[0::3]   # 一次性提取所有 fr
        Qc_arr = local_p[1::3]   # 一次性提取所有 Qc
        phi_arr = local_p[2::3]  # 一次性提取所有 phi
        
        inv_qi = 1.0 / qi_shared
        # 矩阵化防止除 0
        safe_Qc = np.maximum(Qc_arr, 1e-10)
        inv_qc_real = np.cos(phi_arr) / safe_Qc
        
        inv_ql = np.maximum(inv_qi + inv_qc_real, 1e-10)
        Ql_arr = 1.0 / inv_ql
        
        # 将 [fr, Ql, Qc, phi] 拼成 (N, 4) 矩阵并展平
        model_res_params = np.column_stack((fr_arr, Ql_arr, Qc_arr, phi_arr)).ravel()
            
        S_model = s21_model_total(f, model_res_params, base_p, n_res)
        resid_complex = (S_data - S_model)
        return np.hstack([resid_complex.real, resid_complex.imag]) * np.hstack([w_sqrt, w_sqrt])

    if verbose: 
        print(f"  [Shared Qi Fit] 启动归一化优化 (Linear Loss)...")
        print(f"  -> 初始 Shared Qi: {qi_shared_init:.0f} (允许波动范围: {qi_min:.0f} ~ {qi_max:.0f})")
    
  # 3. 极速收敛配置
    res = least_squares(
        fun, x0_norm, bounds=bounds_norm, 
        jac='2-point', 
        loss='linear', 
        x_scale='jac', 
        # ==========================================
        # 【提速核心】将 tol 从 1e-8 放宽到 1e-5 (实验数据最优解)
        # ==========================================
        xtol=1e-5, ftol=1e-5, gtol=1e-5,  
        max_nfev=200,    # 200步上限完全足够
        verbose=2 if verbose else 0
    )
    
    final_real = res.x * scales_arr
    final_base = final_real[:3]
    final_qi = final_real[3] 
    final_local = final_real[4:]
    
    resonators = []
    final_res_p = []
    
    for i in range(n_res):
        fr, Qc, phi = final_local[i*3 : (i+1)*3]
        inv_qi = 1.0 / final_qi
        inv_qc_real = np.cos(phi) / Qc if Qc > 0 else 0.0
        inv_ql = max(inv_qi + inv_qc_real, 1e-10)
        Ql = 1.0 / inv_ql
        k = Ql / Qc if Qc > 0 else 0.0
        
        resonators.append({'fr': fr, 'Ql': Ql, 'Qc': Qc, 'Qi': final_qi, 'phi': phi, 'k': k})
        final_res_p.extend([fr, Ql, Qc, phi])
        
    S_fit = s21_model_total(f, np.array(final_res_p), final_base, n_res)
    
    if verbose:
        print(f"  -> 优化完成! 最终 Shared Qi: {final_qi:.0f} (改变量: {(final_qi-qi_shared_init)/qi_shared_init*100:.1f}%)")
    
    final_resid_abs2 = np.abs(S_data - S_fit)**2
    eval_weights = weights if weights is not None else np.ones_like(S_data)
    weighted_residual = np.sqrt(np.average(final_resid_abs2, weights=eval_weights))
    
    return {
        'resonators': resonators,
        'S_fit': S_fit,
        'baseline_params': final_base,
        'residual': weighted_residual,    # <== 使用加权计算结果
        'scheme_name': 'Global_Hard_SharedQi_Reparam_Normalized'
    }

# ==================== 主拟合流程 ====================

def hierarchical_fit_s21_notch_multi(
    f: np.ndarray, S: np.ndarray, 
    params: Optional[AslsParams] = None,
    tau_plot: bool = False, 
    preview_plot: bool = False,
    prominence_threshold: float = CFG_PROMINENCE, 
    height_threshold: float = CFG_HEIGHT,
    distance: int = CFG_DISTANCE, 
    min_Q: float = CFG_MIN_Q, 
    linewidth_factor: float = CFG_LW_FACTOR,
    min_window_width_hz: float = CFG_MIN_WIN_HZ,
    min_pts_per_win: int = CFG_MIN_PTS_WIN,
    plot: bool = False,
    window_plot: bool = False,
    verbose: bool = True, 
    use_weighting: bool = CFG_USE_WEIGHT, 
    weighting_sigma_factor: float = CFG_WEIGHT_SIGMA,
    enable_downsample: bool = CFG_DOWNSAMPLE
) -> FitOutput:
    
    if params is None: params = AslsParams()
    
    if verbose:
        print("="*60)
        print("开始 S21 多谐振器全局拟合 (Ultimate Edition)")
        print("="*60)
    
    # 1. 预处理
    if verbose: print("\n>>> Step 1: 信号校准 (粗调)...")
    tau0 = estimate_tau_robust(f, S, plot=tau_plot)
    S_tau = S * np.exp(1j * PI_2 * f * tau0)
    
    f_sort, S_flat, A_hat, base_db, base_ph = amplitude_baseline_normalize(
        f, S_tau, params=params, plot=preview_plot
    )
    
    # 2. 检测谐振窗口
    if verbose: print("\n>>> Step 2: 谐振窗口检测...")
    windows = build_resonance_windows(
        f_sort, S_flat, 
        prominence_threshold=prominence_threshold,
        height_threshold=height_threshold, 
        distance=distance, 
        min_Q=min_Q,
        linewidth_factor=linewidth_factor, 
        min_window_width_hz=min_window_width_hz,
        plot=window_plot
    )
    
    # 3. 独立拟合
    if verbose: print("\n>>> Step 3: 独立拟合 (局部精细检测)...")
    indep_results = []
    window_info = []
    
    from scipy.signal import find_peaks, peak_widths
    
    for i, (l, r) in enumerate(windows):
        mask = (f_sort >= l) & (f_sort <= r)
        f_win, S_win = f_sort[mask], S_flat[mask]
        
        if len(f_win) < max(5, min_pts_per_win): 
            window_info.append({'window':(l,r), 'status': 'too_few_points'})
            continue
            
        mag_db_win = 20 * np.log10(np.abs(S_win) + 1e-20)
        pks, _ = find_peaks(-mag_db_win, prominence=prominence_threshold, height=-height_threshold, distance=20)
        
        valid_pks = []
        if len(pks) > 0:
            widths, _, left_ips, right_ips = peak_widths(-mag_db_win, pks, rel_height=0.5)
            left_ips = np.clip(left_ips, 0, len(f_win)-1)
            right_ips = np.clip(right_ips, 0, len(f_win)-1)
            f_left = np.interp(left_ips, np.arange(len(f_win)), f_win)
            f_right = np.interp(right_ips, np.arange(len(f_win)), f_win)
            bw_hz = f_right - f_left
            f0s = f_win[pks]
            Q_ests = f0s / np.maximum(bw_hz, 1e-10)
            for k, pk_idx in enumerate(pks):
                if 3 <= pk_idx <= len(f_win) - 4 and min_Q <= Q_ests[k] <= 1e8:
                    valid_pks.append(pk_idx)
        
        if len(valid_pks) == 0: valid_pks = [np.argmin(mag_db_win)]
            
        for pk_idx in valid_pks:
            fr_guess = f_win[pk_idx]
            span = len(f_win) if len(valid_pks)==1 else max(10, len(f_win) // (len(valid_pks) * 2))
            s_idx = max(0, pk_idx - span)
            e_idx = min(len(f_win), pk_idx + span)
            
            f_sub, S_sub = f_win[s_idx:e_idx], S_win[s_idx:e_idx]
            if len(f_sub) < 5: continue
            
            try:
                fit_res = fit_single_notch_local(f_sub, S_sub, plot=False)
                if not (f_sub[0] <= fit_res['fr'] <= f_sub[-1] and 0 <= fit_res['Ql'] <= 1e9):
                    raise ValueError("Fit out of bounds")
                indep_results.append(fit_res)
                window_info.append({'window':(l,r), 'status':'ok'})
            except:
                pass
            
    if not indep_results: raise RuntimeError("无法提取任何谐振器初值！")
    if verbose: print(f"Step 3 完成: 提取 {len(indep_results)} 个有效谐振峰")
        
    # 4. 降采样
    f_opt, S_opt = f_sort, S_flat
    ds_info = None
    if enable_downsample:
        f_opt, S_opt, _ = _adaptive_downsample_with_windows(f_sort, S_flat, windows, indep_results, verbose)
        ds_info = {'orig': len(f_sort), 'new': len(f_opt)}
        
    # 5. 全局优化
    if verbose: print("\n>>> Step 4: 全局联合优化...")
    
    # ��已修改】将基线的权重极度压低，让峰值区域绝对主导 (100倍差异)
    weights = np.ones_like(f_opt) * 0.0001
    if use_weighting:
        for res in indep_results:
            fr, Ql = res['fr'], res['Ql']
            sigma = fr / Ql * weighting_sigma_factor
            w = np.exp(-0.5 * ((f_opt - fr)/sigma)**2)
            weights = np.maximum(weights, w)
            
    g_res = _optimize_global_robust(
        f_opt, S_opt, indep_results, 
        verbose=verbose, weights=weights,
        qi_bound_ratio=CFG_QI_BOUND_RATIO,
        qc_bound_ratio=CFG_QC_BOUND_RATIO
    )
    
    final_results = g_res['resonators']
    final_base_p = g_res['baseline_params']
    
    # 6. 重建结果
    if verbose: print("\n>>> Step 5: 重建最终结果...")
    final_res_p = []
    for res in final_results:
        final_res_p.extend([res['fr'], res['Ql'], res['Qc'], res['phi']])
        
    S_fit_cal_full = s21_model_total(f_sort, np.array(final_res_p), final_base_p, len(final_results))
    S_fit_final = S_fit_cal_full * A_hat * np.exp(1j * base_ph) * np.exp(-1j * PI_2 * f_sort * tau0)
    
    output = {
        'global_params': {'tau': tau0, 'f_range': [f_sort.min(), f_sort.max()]},
        'resonators': final_results,
        'window_info': window_info,
        'S21_fit': S_fit_final,
        'S21_fit_cal': S_fit_cal_full,
        'optimization_scheme': g_res['scheme_name'],
        'downsample_enabled': enable_downsample,
        'weighting_enabled': use_weighting,
        'downsample_info': ds_info
    }
    
    if plot:
        plot_fit_results(f_sort, S, output, r'global_robust_fit.png')
        
    return output

# ==================== 绘图函数 ====================

def plot_fit_results(f: np.ndarray, S: np.ndarray, result: Dict, save_path: Optional[str] = None):
    S_fit = result['S21_fit']
    resid = S - S_fit
    resid_abs = np.abs(resid)
    
    # ========================================================
    # 【新增逻辑】：计算加权 RMS，将评估重心强行锁定在谐振坑底
    # ========================================================
    weights = np.ones_like(f) * 0.01  # 基线权重仅给 0.01
    for res in result['resonators']:
        fr, Ql = res['fr'], res['Ql']
        sigma = fr / Ql * CFG_WEIGHT_SIGMA
        w = np.exp(-0.5 * ((f - fr)/sigma)**2)
        weights = np.maximum(weights, w)
        
    # 计算加权 RMS (Weighted RMS)
    weighted_rms = np.sqrt(np.average(resid_abs**2, weights=weights))
    # 计算传统全局 RMS 留作对比
    global_rms = np.sqrt(np.mean(resid_abs**2))
    # ========================================================
    
    fig, axes = plt.subplots(4, 1, figsize=(7, 12), constrained_layout=True)
    
    ax = axes[0]
    ax.plot(f/1e9, 20*np.log10(np.abs(S)), 'b-', alpha=0.5, lw=1, label='Data')
    ax.plot(f/1e9, 20*np.log10(np.abs(S_fit)), 'r--', lw=1.5, label='Fit')
    ax.set_ylabel('|S21| (dB)'); ax.set_title('(a) Magnitude')
    ax.legend(); ax.grid(alpha=0.3)
    
    ax = axes[1]
    ax.plot(f/1e9, np.angle(S), 'b-', alpha=0.5, lw=1)
    ax.plot(f/1e9, np.angle(S_fit), 'r--', lw=1.5)
    ax.set_ylabel('Phase (rad)'); ax.set_title('(b) Phase')
    ax.grid(alpha=0.3)
    
    ax = axes[2]
    ax.plot(S.real, S.imag, 'b-', alpha=0.3, lw=1)
    ax.plot(S_fit.real, S_fit.imag, 'r--', lw=1.5)
    ax.set_xlabel('Re'); ax.set_ylabel('Im'); ax.set_title('(c) IQ Plane')
    ax.axis('equal'); ax.grid(alpha=0.3)
    
    # 残差图
    ax = axes[3]
    ax.plot(f/1e9, resid_abs, 'k-', lw=1, alpha=0.7, label='Abs Residual')
    
    # 【可视化强化】用浅绿色背景标出高权重的评估区域 (谐振坑)
    ax2 = ax.twinx()
    ax2.fill_between(f/1e9, 0, weights, color='green', alpha=0.15, label='Evaluation Weight')
    ax2.set_ylim(0, 1.1)
    ax2.set_yticks([])  # 隐藏副坐标轴的刻度
    
    ax.set_ylabel('Resid Abs'); ax.set_xlabel('Freq (GHz)')
    # 在标题同时展示两个 RMS，你会发现 Weighted RMS 更能真实反映坑底拟合度！
    ax.set_title(f'(d) Residual (Weighted RMS = {weighted_rms:.2e} | Global RMS = {global_rms:.2e})')
    ax.grid(alpha=0.3)
    
    if save_path: plt.savefig(save_path, dpi=300)
    plt.show()

# ==================== 主入口 ====================

def main():
    import time
    start_time = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=str, nargs='?', default=DEFAULT_FILE)
    parser.add_argument("--no-plot", action="store_true", help="禁用绘图")
    args = parser.parse_args()
    
    if not os.path.exists(args.file_path):
        print(f"File not found: {args.file_path}"); return

    try:
        if args.file_path.endswith('.s2p'):
            f, S = read_s2p(args.file_path)
        else:
            f, S = read_txt(args.file_path)
            
        print(f"Data Loaded: {len(f)} pts, {f.min()/1e9:.3f}-{f.max()/1e9:.3f} GHz")
        
        params = AslsParams()
        do_plot = CFG_PLOT_FINAL if not args.no_plot else False
        
        res = hierarchical_fit_s21_notch_multi(
            f, S, params=params,
            tau_plot=CFG_PLOT_TAU,
            preview_plot=CFG_PLOT_PREVIEW,
            window_plot=CFG_PLOT_WINDOWS,
            enable_downsample=CFG_DOWNSAMPLE,
            plot=do_plot,
            verbose=CFG_VERBOSE
        )
        
        print("\n" + "="*70)
        print("Final Fitting Results (Ultimate Edition)")
        print("="*70)
        # 【修改】控制台打印加上高精度的 Phi
        print(f"{'Idx':<4} {'Freq(GHz)':<12} {'Ql':<10} {'Qi':<10} {'Qc':<10} {'Phi(rad)':<10}")
        print("-" * 70)
        
        results_list = []
        for i, r in enumerate(res['resonators']):
            print(f"{i+1:<4} {r['fr']/1e9:<12.6f} {r['Ql']:<10.0f} {r['Qi']:<10.0f} "
                  f"{r['Qc']:<10.0f} {r['phi']:<10.5f}")
            
            results_list.append([
                i+1, r['fr']/1e9, r['Ql'], r['Qi'], r['Qc'], r['phi'], r['k']
            ])
            
        save_dir = os.path.dirname(args.file_path)
        csv_name = f"fit_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        csv_path = os.path.join(save_dir, csv_name)
        
        df = pd.DataFrame(results_list, columns=['Index', 'Freq_GHz', 'Ql', 'Qi', 'Qc', 'Phi', 'k'])
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback; traceback.print_exc()
        
    print(f"\nTotal Time: {time.time()-start_time:.2f}s")

if __name__ == "__main__":
    main()