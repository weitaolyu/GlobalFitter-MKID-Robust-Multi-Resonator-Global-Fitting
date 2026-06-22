# -*- coding: utf-8 -*-
"""
MKID/超导谐振器 S21 全局拟合算法 (Soft shared Qi 版)
包含:
1. 动态基线优化
2. 归一化全局优化
3. Soft shared Qi（允许每个峰有少量 Qi 不一致）
Created on: 2026-03-02
Updated: 2026-04-22
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import warnings
import argparse
from typing import Dict, List, Optional, NamedTuple, Any

try:
    from typing import TypedDict
except ImportError:
    from typing import Dict as TypedDict

from scipy.optimize import least_squares

# =========================================================================================
#                                【 用户可调参数配置区 】
# =========================================================================================

# ----------------- 1. 文件与基础配置 -----------------
DEFAULT_FILE = r'S21_simulation.txt'

# ----------------- 2. 谐振峰检测 (Peak Detection) -----------------
CFG_DISTANCE = 30
CFG_MIN_Q = 1000.0
CFG_PROMINENCE = 10       
CFG_HEIGHT = -10.0       
CFG_MIN_WIN_HZ = 500e3     
CFG_LW_FACTOR = 10.0  

# CFG_DISTANCE = 5
# CFG_MIN_Q = 1000.0
# CFG_PROMINENCE = 3      
# CFG_HEIGHT = -3       
# CFG_MIN_WIN_HZ = 70e3     
# CFG_LW_FACTOR = 3.0  

# ----------------- 3. 优化算法控制 (Optimization Control) -----------------
CFG_USE_WEIGHT = True  # 是否对频率点使用加权拟合
CFG_WEIGHT_SIGMA = 3.0 # 权重函数宽度（通常以线宽归一化后的 sigma 表示）
CFG_QI_BOUND_RATIO = 0.5   # shared Qi center 相对初始中位数允许浮动的比例

# ----------------- 3.1 Soft shared Qi 控制 -----------------
CFG_SOFT_QI_SIGMA_LOG = 0.05      # log(Qi) 的 1σ 离散（约等于 5%）
CFG_SOFT_QI_WEIGHT = 0.1          # soft penalty 强度
CFG_SOFT_QI_MAX_DELTA_SIGMA = 15.0 # 每个峰允许偏离中心的最大 sigma 数

# ----------------- 4. 降采样策略 (Downsampling) -----------------
CFG_DOWNSAMPLE = True

# ----------------- 5. 基线校准算法 (ASLS Baseline) -----------------
CFG_ASLS_LAM = 1e6
CFG_ASLS_P_UP = 0.05
CFG_PHASE_LAM = 1e6
CFG_PHASE_P = 0.5

# ----------------- 6. 可视化与调试 (Plot & Debug) -----------------
CFG_PLOT_FINAL = True
CFG_VERBOSE = True

# =========================================================================================

# ==================== 依赖检查 ====================
try:
    from main_data_reader import read_s2p, read_txt
    from main_estimate_delay import estimate_tau_robust
    from main_amplitude_baseline_normalize import amplitude_baseline_normalize
    from main_build_resonance_windows_v1 import build_resonance_windows
    from main_single_resonator_v1 import fit_single_notch_local
except ImportError as e:
    print(f"[严重错误] 缺失库: {e}")
    sys.exit(1)

# ==================== 配置 ====================
PI_2 = 2.0 * np.pi
EPS = 1e-18

def configure_plotting():
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['figure.dpi'] = 600

configure_plotting()
warnings.filterwarnings('ignore', category=UserWarning)

# ==================== 类型定义 ====================

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
    optimization_scheme: str
    downsample_info: Optional[Dict]

# =========================================================================================
# ==================== 核心物理模型 ====================
# =========================================================================================

def s21_model_total(
    f: np.ndarray,
    res_params_flat: np.ndarray,
    baseline_params: np.ndarray,
    n_res: int
) -> np.ndarray:
    """
    包含动态基线和残余延时的完整 S21 模型 (向量化)

    res_params_flat:
        [fr1, Ql1, Qc1, phi1, fr2, Ql2, Qc2, phi2, ...]
    baseline_params:
        [A, alpha, tau_residual]
    """
    # 1) 基线参数
    A, alpha, tau = baseline_params
    S_baseline = A * np.exp(1j * (alpha + PI_2 * f * tau))

    # 2) 谐振器参数解析
    p = res_params_flat.reshape(n_res, 4)
    fr = p[:, 0]
    Ql = p[:, 1]
    Qc = p[:, 2]
    phi = p[:, 3]

    # 3) 计算 resonator 乘积
    k = np.zeros_like(Ql)
    mask = np.abs(Qc) > 1e-9
    k[mask] = Ql[mask] / Qc[mask]

    x = (f[:, None] - fr) / fr
    denom = 1.0 + 2j * Ql * x
    term = k * np.exp(1j * phi) / denom
    S_res = np.prod(1.0 - term, axis=1)

    return S_res * S_baseline


def _extract_resonator_arrays(results: List[Dict]) -> np.ndarray:
    """
    返回:
        [[fr, Ql, phi, k], ...]
    """
    return np.array([[res['fr'], res['Ql'], res['phi'], res['k']] for res in results])


def _calculate_qi_from_params(fr, Ql, phi, k):
    """
    与当前全局优化器保持一致的 Qi / Qc 关系：

        k = Ql / Qc
        1/Qi = 1/Ql - cos(phi)/Qc
    """
    if abs(k) > 1e-9:
        Qc = Ql / k
        inv_qi = 1.0 / Ql - np.cos(phi) / Qc
        if inv_qi > 1e-10:
            Qi = 1.0 / inv_qi
        else:
            Qi = 1e8
    else:
        Qc, Qi = 1e9, Ql
    return Qi, Qc

# =========================================================================================
# ==================== 工具函数 ====================
# =========================================================================================

def _clip_inside_bounds(x, lo, hi, eps_frac=1e-8):
    """
    将初值 x 强制压到 bounds 内部，避免 least_squares 因 x0 越界直接报错。
    """
    x = np.asarray(x, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)

    span = np.maximum(hi - lo, 1.0)
    x = np.maximum(x, lo + eps_frac * span)
    x = np.minimum(x, hi - eps_frac * span)
    return x

# =========================================================================================
# ==================== 降采样逻辑 ====================
# =========================================================================================

def _adaptive_downsample_with_windows(f, S, windows, indep_results, verbose=True):
    n_points = len(f)
    keep_indices = set()

    for i, (l, r) in enumerate(windows):
        idx = np.where((f >= l) & (f <= r))[0]
        if len(idx) == 0:
            continue

        # 每个窗口限制点数
        step = max(1, len(idx) // 200)
        keep_indices.update(idx[::step])

    # 加上一些全局基线点
    all_idx = np.arange(0, n_points, max(1, n_points // 500))
    keep_indices.update(all_idx)

    keep_indices = sorted(list(keep_indices))
    if verbose:
        print(f"  降采样: {n_points} -> {len(keep_indices)}")

    keep_indices = np.array(keep_indices, dtype=int)
    return f[keep_indices], S[keep_indices], keep_indices

# =========================================================================================
# ==================== 全局优化器（Soft shared Qi） ====================
# =========================================================================================

def _optimize_global_robust(
    f: np.ndarray,
    S_data: np.ndarray,
    independent_results: List[Dict],
    verbose: bool = True,
    weights: Optional[np.ndarray] = None,
    qi_bound_ratio: float = CFG_QI_BOUND_RATIO,
    sigma_log_qi: float = CFG_SOFT_QI_SIGMA_LOG,
    soft_qi_weight: float = CFG_SOFT_QI_WEIGHT,
    max_delta_sigma: float = CFG_SOFT_QI_MAX_DELTA_SIGMA
) -> Dict:
    """
    归一化优化 + Soft shared Qi

    全局参数:
        [A, alpha, tau_residual, logQi_center]

    每个谐振器局部参数:
        [fr, Qc, phi, dlogQi]

    其中:
        logQi_i = logQi_center + dlogQi

    并对 dlogQi 加 soft penalty:
        penalty_i = soft_qi_weight * dlogQi / sigma_log_qi
    """
    n_res = len(independent_results)
    res_arrays = _extract_resonator_arrays(independent_results)

    # ------------------------------------------------------------
    # 1. 初始 Qi_center
    # ------------------------------------------------------------
    qi_vals = []
    for row in res_arrays:
        qi, _ = _calculate_qi_from_params(*row)
        if 1e3 < qi < 1e9:
            qi_vals.append(qi)

    qi_center_init = np.median(qi_vals) if qi_vals else 1e5
    log_qi_center_init = np.log(qi_center_init)

    # ------------------------------------------------------------
    # 2. 全局参数:
    #    [A, alpha, tau, logQi_center]
    # ------------------------------------------------------------
    p_global = [1.0, 0.0, 0.0, log_qi_center_init]
    scales = [1.0, 1.0, 1e-9, 1.0]

    qi_min = max(qi_center_init * (1.0 - qi_bound_ratio), 1e3)
    qi_max = qi_center_init * (1.0 + qi_bound_ratio)

    bounds_low = [0.8, -np.pi, -1e-8, np.log(qi_min)]
    bounds_high = [1.2, np.pi,  1e-8, np.log(qi_max)]

    # ------------------------------------------------------------
    # 3. 每个谐振器局部参数:
    #    [fr, Qc, phi, dlogQi]
    # ------------------------------------------------------------
    p_local = []

    dlog_lo = -max_delta_sigma * sigma_log_qi
    dlog_hi = +max_delta_sigma * sigma_log_qi

    for i in range(n_res):
        fr, Ql, phi, k = res_arrays[i]
        qi_est, Qc = _calculate_qi_from_params(fr, Ql, phi, k)

        p_local.extend([fr, Qc, phi, 0.0])   # 初始 dlogQi = 0

        scales.extend([
            fr,
            max(Qc, 1e3),
            max(abs(phi), 1.0),
            max(sigma_log_qi, 1e-3)
        ])

        bw = fr / max(Ql, 1.0)
        fr_margin = bw * 0.002  # 保持你原来的 2% 线宽锁定

        bounds_low.extend([
            fr - fr_margin,
            max(Qc * 0.1, 1e2),
            phi - np.pi,
            dlog_lo
        ])
        bounds_high.extend([
            fr + fr_margin,
            max(Qc * 10.0, 1e3),
            phi + np.pi,
            dlog_hi
        ])

    x0_real = np.array(p_global + p_local, dtype=float)
    bounds_low_real = np.array(bounds_low, dtype=float)
    bounds_high_real = np.array(bounds_high, dtype=float)
    scales_arr = np.array(scales, dtype=float)

    # 初值压进 bounds，避免 x0 越界
    x0_real = _clip_inside_bounds(x0_real, bounds_low_real, bounds_high_real)

    # ------------------------------------------------------------
    # 4. 彻底归一化
    # ------------------------------------------------------------
    x0_norm = x0_real / scales_arr
    bounds_norm = (bounds_low_real / scales_arr, bounds_high_real / scales_arr)

    w_sqrt = np.sqrt(weights) if weights is not None else 1.0

    # ------------------------------------------------------------
    # 5. 残差函数：复数残差 + soft Qi penalty
    # ------------------------------------------------------------
    def fun(p_norm):
        p_real = p_norm * scales_arr

        base_p = p_real[:3]              # [A, alpha, tau]
        log_qi_center = p_real[3]
        local_p = p_real[4:]

        model_res_params = []
        soft_qi_resid = []

        for i in range(n_res):
            fr, Qc, phi, dlogQi = local_p[i*4:(i+1)*4]

            log_qi_i = log_qi_center + dlogQi
            Qi_i = np.exp(log_qi_i)

            # 1/Ql = 1/Qi + cos(phi)/Qc
            inv_qi = 1.0 / Qi_i
            inv_qc_real = np.cos(phi) / Qc if Qc > 0 else 0.0
            inv_ql = max(inv_qi + inv_qc_real, 1e-10)
            Ql = 1.0 / inv_ql

            model_res_params.extend([fr, Ql, Qc, phi])

            # soft shared Qi penalty
            resid_qi_i = soft_qi_weight * (dlogQi / sigma_log_qi)
            soft_qi_resid.append(resid_qi_i)

        S_model = s21_model_total(f, np.array(model_res_params), base_p, n_res)
        resid_complex = (S_data - S_model)
        resid_data = np.hstack([resid_complex.real, resid_complex.imag]) * np.hstack([w_sqrt, w_sqrt])

        return np.hstack([resid_data, np.array(soft_qi_resid)])

    if verbose:
        print("  [Soft Shared Qi Fit] 启动归一化优化...")
        print(f"  -> 初始 Qi_center: {qi_center_init:.0f}")
        print(f"  -> Soft Qi sigma(log): {sigma_log_qi:.3f}, weight: {soft_qi_weight:.2f}")
        print(f"  -> Qi_center 允许范围: {qi_min:.0f} ~ {qi_max:.0f}")
        print(f"  -> 每个峰 dlogQi 允许范围: [{dlog_lo:.3f}, {dlog_hi:.3f}]")

    res = least_squares(
        fun, x0_norm, bounds=bounds_norm,
        jac='3-point',
        loss='linear',
        x_scale='jac',
        xtol=1e-12, ftol=1e-12, gtol=1e-12,
        max_nfev=300,
        verbose=2 if verbose else 0
    )

    # ------------------------------------------------------------
    # 6. 结果还原
    # ------------------------------------------------------------
    final_real = res.x * scales_arr

    final_base = final_real[:3]
    final_log_qi_center = final_real[3]
    final_qi_center = np.exp(final_log_qi_center)
    final_local = final_real[4:]

    resonators = []
    final_res_p = []
    qi_list_final = []
    dlog_list_final = []

    for i in range(n_res):
        fr, Qc, phi, dlogQi = final_local[i*4:(i+1)*4]

        log_qi_i = final_log_qi_center + dlogQi
        Qi_i = np.exp(log_qi_i)

        inv_qi = 1.0 / Qi_i
        inv_qc_real = np.cos(phi) / Qc if Qc > 0 else 0.0
        inv_ql = max(inv_qi + inv_qc_real, 1e-10)
        Ql = 1.0 / inv_ql

        k = Ql / Qc if Qc > 0 else 0.0

        resonators.append({
            'fr': fr,
            'Ql': Ql,
            'Qc': Qc,
            'Qi': Qi_i,
            'phi': phi,
            'k': k,
            'dlogQi': dlogQi
        })

        qi_list_final.append(Qi_i)
        dlog_list_final.append(dlogQi)
        final_res_p.extend([fr, Ql, Qc, phi])

    S_fit_final = s21_model_total(f, np.array(final_res_p), final_base, n_res)

    if verbose:
        qi_arr = np.array(qi_list_final)
        dlog_arr = np.array(dlog_list_final)
        print(f"  -> 优化完成! Soft Qi center: {final_qi_center:.0f}")
        print(f"  -> 各 resonator Qi: mean={np.mean(qi_arr):.0f}, std={np.std(qi_arr):.0f}")
        print(f"  -> 各 resonator dlogQi: {np.array2string(dlog_arr, precision=4, suppress_small=False)}")
        print(
            f"  -> 基线参数: "
            f"A={final_base[0]:.6f}, "
            f"alpha={final_base[1]:.6f}, "
            f"tau_residual={final_base[2]:.3e}"
        )

    return {
        'resonators': resonators,
        'S_fit': S_fit_final,
        'baseline_params': final_base,
        'qi_center_soft': final_qi_center,
        'residual': np.sqrt(np.mean(np.abs(S_data - S_fit_final)**2)),
        'scheme_name': 'Robust_Global_Soft_Shared_Qi_Normalized'
    }

# =========================================================================================
# ==================== 主流程 ====================
# =========================================================================================

def hierarchical_fit_s21_notch_multi(
    f: np.ndarray, S: np.ndarray,
    params: Optional[AslsParams] = None,
    enable_downsample: bool = CFG_DOWNSAMPLE,
    use_weighting: bool = CFG_USE_WEIGHT,
    verbose: bool = CFG_VERBOSE,
    plot_result: bool = CFG_PLOT_FINAL,
    save_path: Optional[str] = None
) -> FitOutput:

    if params is None:
        params = AslsParams()

    # 1. 预处理 (粗基线)
    if verbose:
        print(">>> Step 1: 粗基线去除...")
    tau0 = estimate_tau_robust(f, S, plot=False)
    S_tau = S * np.exp(1j * PI_2 * f * tau0)
    f_sort, S_flat, A_hat, base_db, base_ph = amplitude_baseline_normalize(f, S_tau, params=params)

    # 2. 找峰
    if verbose:
        print(">>> Step 2: 谐振峰检测...")
    # windows = build_resonance_windows(f_sort, S_flat, min_Q=CFG_MIN_Q, distance=CFG_DISTANCE)
    windows = build_resonance_windows(
        f_sort, S_flat, 
        prominence_threshold=CFG_PROMINENCE,
        height_threshold=CFG_HEIGHT, 
        distance=CFG_DISTANCE, 
        min_Q=CFG_MIN_Q,
        linewidth_factor=CFG_LW_FACTOR, 
        min_window_width_hz=CFG_MIN_WIN_HZ
    )
    if verbose:
        print(f"    检测到 {len(windows)} 个谐振模式")

    # 3. 独立拟合 (初值)
    if verbose:
        print(">>> Step 3: 独立拟合 (初值)...")
    indep_results = []
    win_info = []

    for i, (l, r) in enumerate(windows):
        mask = (f_sort >= l) & (f_sort <= r)
        try:
            res = fit_single_notch_local(f_sort[mask], S_flat[mask])
            indep_results.append(res)
            win_info.append({'window': (l, r), 'status': 'ok'})

            if verbose:
                fr_val = res['fr']
                Ql_val = res['Ql']
                phi_val = res['phi']
                k_val = res['k']
                Qi_val, Qc_val = _calculate_qi_from_params(fr_val, Ql_val, phi_val, k_val)
                print(
                    f"    [初值 {len(indep_results)}] "
                    f"fr: {fr_val/1e9:.6f} GHz, "
                    f"Ql: {Ql_val:.0f}, "
                    f"Qc: {Qc_val:.0f}, "
                    f"Qi: {Qi_val:.0f}, "
                    f"phi: {phi_val:.3f}"
                )

        except Exception as e:
            if verbose:
                print(f"    [初值] 窗口 {i+1} 拟合失败: {e}")

    if not indep_results:
        raise RuntimeError("无有效谐振器")

    # 4. 准备全局数据
    f_opt, S_opt = f_sort, S_flat
    ds_info = None
    if enable_downsample:
        f_opt, S_opt, idxs = _adaptive_downsample_with_windows(
            f_sort, S_flat, windows, indep_results, verbose
        )
        ds_info = {'orig': len(f_sort), 'new': len(f_opt)}

    # 5. 全局优化
    if verbose:
        print(">>> Step 4: 全局优化 (Robust Mode + Soft shared Qi)...")

    weights = np.ones_like(f_opt) * 0.1
    if use_weighting:
        for r in indep_results:
            w = np.exp(-0.5 * ((f_opt - r['fr']) / (r['fr'] / r['Ql'] * CFG_WEIGHT_SIGMA))**2)
            weights = np.maximum(weights, w)

    g_res = _optimize_global_robust(
        f_opt, S_opt, indep_results,
        verbose=verbose,
        weights=weights,
        qi_bound_ratio=CFG_QI_BOUND_RATIO,
        sigma_log_qi=CFG_SOFT_QI_SIGMA_LOG,
        soft_qi_weight=CFG_SOFT_QI_WEIGHT,
        max_delta_sigma=CFG_SOFT_QI_MAX_DELTA_SIGMA
    )

    # 6. 重建最终结果 (还原到原始数据尺度)
    final_res_p = []
    for r in g_res['resonators']:
        final_res_p.extend([r['fr'], r['Ql'], r['Qc'], r['phi']])
    final_base_p = g_res['baseline_params']

    S_fit_flat_full = s21_model_total(f_sort, np.array(final_res_p), final_base_p, len(indep_results))

    # 加上之前的粗基线和延时
    S_final = S_fit_flat_full * A_hat * np.exp(1j * base_ph) * np.exp(-1j * PI_2 * f_sort * tau0)

    output: FitOutput = {
        'resonators': g_res['resonators'],
        'S21_fit': S_final,
        'global_params': {
            'tau': tau0,
            'base_db': base_db,
            'baseline_params': g_res['baseline_params'],
            'qi_center_soft': g_res['qi_center_soft']
        },
        'window_info': win_info,
        'optimization_scheme': 'Robust_Global_Soft_Qi',
        'downsample_info': ds_info
    }

    if plot_result:
        
        plot_fit_results(f_sort, S, output, save_path= r'C:\Users\NEVER\Desktop\HK\ROGer\usual code\S21fitting\20260423\global.png')

    return output

# =========================================================================================
# ==================== 绘图与入口 ====================
# =========================================================================================

def plot_fit_results(f, S, result, save_path=None):
    import numpy as np
    import matplotlib.pyplot as plt

    S_fit = result['S21_fit']
    f_GHz = f / 1e9

    # =========================
    # 统一配色（与前图保持一致）
    # =========================
    data_color = '#0000FF'   # 蓝：原始数据
    fit_color  = '#FFA500'   # 橙：拟合结果
    mark_color = '#008000'   # 绿：参考/标记（这里可少量使用）

    # 误差颜色（更协调、更好看）

    mag_err_color   = '#00FFFF'   # pure cyan
    phase_err_color = '#FF00FF'   # pure magenta


    # ========= 辅助函数：unwrap + 去线性趋势 =========
    def unwrap_and_detrend_phase(freq, S_complex):
        phase = np.unwrap(np.angle(S_complex))
        p = np.polyfit(freq, phase, 1)
        trend = np.polyval(p, freq)
        phase_detrended = phase - trend
        return phase_detrended, phase, trend

    # =========================
    # 误差计算
    # =========================
    mag_err = np.abs(S) - np.abs(S_fit)

    # wrap 到 [-pi, pi]
    phase_err = np.angle(np.exp(1j * (np.angle(S) - np.angle(S_fit))))

    # 去线性趋势后的 unwrapped phase
    phase_dt, phase_unwrap, trend_data = unwrap_and_detrend_phase(f, S)
    phase_fit_dt, phase_fit_unwrap, trend_fit = unwrap_and_detrend_phase(f, S_fit)

    # RMS
    mag_rms = np.sqrt(np.mean(mag_err**2))
    phase_rms = np.sqrt(np.mean(phase_err**2))

    # =========================
    # 统一论文风格参数
    # =========================
    plt.rcParams.update({

        'font.size': 15,
        'axes.labelsize': 16,
        'axes.titlesize': 17,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 13,

        'lines.linewidth': 2.0,
        'font.family': 'serif',
        'mathtext.fontset': 'stix'
    })

    # 4行1列
    fig, axes = plt.subplots(4, 1, figsize=(13, 13), sharex=True)

    # ======================================================
    # (a) 幅度拟合
    # ======================================================
    ax = axes[0]
    ax.plot(
        f_GHz, 20*np.log10(np.abs(S)),
        color=data_color, ls='-', lw=3, label='Raw amplitude'
    )
    ax.plot(
        f_GHz, 20*np.log10(np.abs(S_fit)),
        color=fit_color, ls='--', lw=3, label='Global fit'
    )
    ax.set_ylabel(r'$|S_{21}|$ (dB)')
    ax.set_title('(a) $|S_{21}|$ Comparison')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, ls='--', lw=0.6)
    ax.tick_params(direction='in')

    # ======================================================
    # (b) 去线性趋势后的 unwrapped phase 对比
    # ======================================================
    ax = axes[1]
    ax.plot(
        f_GHz, phase_dt,
        color=data_color, ls='-', lw=3, label='Raw detrended phase'
    )
    ax.plot(
        f_GHz, phase_fit_dt,
        color=fit_color, ls='--', lw=3, label='Global fit detrended phase'
    )
    ax.set_ylabel('Detrended phase (rad)')
    ax.set_title('(b) Detrended Unwrapped Phase Comparison')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, ls='--', lw=0.6)
    ax.tick_params(direction='in')

    # ======================================================
    # (c) 幅度误差
    # ======================================================
    ax = axes[2]
    ax.plot(
        f_GHz, mag_err,
        color=mag_err_color, ls='-', lw=3, label='Magnitude error'
    )
    ax.set_ylabel('$|S_{21}|$ error')
    ax.set_title(f'(c) $|S_{{21}}|$ Error  (RMS = {mag_rms:.2e})')
    ax.grid(True, alpha=0.3, ls='--', lw=0.6)
    ax.tick_params(direction='in')

    # 可选：加 0 参考线
    ax.axhline(0, color='0.5', lw=0.8, ls=':', alpha=0.7)

    # ======================================================
    # (d) 相位误差
    # ======================================================
    ax = axes[3]
    ax.plot(
        f_GHz, phase_err,
        color=phase_err_color, ls='-', lw=3, label='Phase error'
    )
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('Phase error (rad)')
    ax.set_title(f'(d) Phase Error  (RMS = {phase_rms:.2e})')
    ax.grid(True, alpha=0.3, ls='--', lw=0.6)
    ax.tick_params(direction='in')

    # 可选：加 0 参考线
    ax.axhline(0, color='0.5', lw=0.8, ls=':', alpha=0.7)

    # =========================
    # 布局优化
    # =========================
    plt.subplots_adjust(
        left=0.11, right=0.97, top=0.96, bottom=0.08,
        hspace=0.28
    )

    if save_path:
        fig.savefig(save_path, dpi=900, bbox_inches='tight')

    plt.show()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=str, nargs='?', default=DEFAULT_FILE)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.file_path):
        print(f"File not found: {args.file_path}")
        return

    try:
        # 读取数据
        if args.file_path.endswith('.s2p'):
            f, S = read_s2p(args.file_path)
        else:
            f, S = read_txt(args.file_path)

        print(f"Data Loaded: {len(f)} pts")

        fit_params = AslsParams()
        do_plot = CFG_PLOT_FINAL if not args.no_plot else False

        res = hierarchical_fit_s21_notch_multi(
            f, S,
            params=fit_params,
            enable_downsample=CFG_DOWNSAMPLE,
            plot_result=do_plot,
            save_path= r'C:\Users\NEVER\Desktop\HK\ROGer\usual code\S21fitting\20260423\global.png'
        )

        print("\n=== Final Results ===")
        print(f"{'Idx':<4} {'Fr(GHz)':<10} {'Qi':<10} {'Qc':<10} {'Ql':<10} {'Phi(rad)':<8} {'dlogQi':<10}")
        for i, r in enumerate(res['resonators']):
            print(
                f"{i+1:<4} "
                f"{r['fr']/1e9:<10.6f} "
                f"{r['Qi']:<10.0f} "
                f"{r['Qc']:<10.0f} "
                f"{r['Ql']:<10.0f} "
                f"{r['phi']:<8.3f} "
                f"{r.get('dlogQi', 0.0):<10.4f}"
            )

        if 'qi_center_soft' in res['global_params']:
            print("\n=== Soft Shared Qi Center ===")
            print(f"Qi_center_soft = {res['global_params']['qi_center_soft']:.0f}")

        if 'baseline_params' in res['global_params']:
            bp = res['global_params']['baseline_params']
            print("\n=== Final Baseline Params ===")
            print(f"A            = {bp[0]:.6f}")
            print(f"alpha        = {bp[1]:.6f}")
            print(f"tau_residual = {bp[2]:.6e}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()