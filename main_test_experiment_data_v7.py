# -*- coding: utf-8 -*-
"""
MKID/超导谐振器 S21 全局拟合算法 (Ultimate Edition, Soft shared Qi)
包含:
1. 动态基线优化
2. Soft shared Qi（允许每个峰有少量 Qi 不一致）
3. 参数归一化
4. cos(phi) 物理修正
5. 加权拟合 + 降采样
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
from scipy.signal import find_peaks, peak_widths

# =========================================================================================
#                                【 用户可调参数配置区 】
# =========================================================================================

# ----------------- 1. 文件与基础配置 -----------------
DEFAULT_FILE = r'100mkoff.s2p'

# ----------------- 2. 谐振峰检测 (Peak Detection) -----------------
CFG_PROMINENCE = 1.5
CFG_HEIGHT = -2.0
CFG_DISTANCE = 20
CFG_MIN_Q = 5000.0

# ----------------- 3. 拟合窗口提取 (Window Building) -----------------
CFG_LW_FACTOR = 5.0
CFG_MIN_WIN_HZ = 500e3
CFG_MIN_PTS_WIN = 20

# ----------------- 4. 优化算法控制 (Optimization Control) -----------------
CFG_USE_WEIGHT = True
CFG_WEIGHT_SIGMA = 3.0

# -------- shared / soft-shared Qi 控制 --------
CFG_QI_BOUND_RATIO = 5   # Qi_center 相对初始中位数允许浮动比例
CFG_QC_BOUND_RATIO = 1.0   # 每个峰的 Qc 相对初值允许变化范围


# -------- Soft shared Qi 控制 --------
CFG_SOFT_QI_SIGMA_LOG = 0.2        # 描述偏离的尺度。这个值越小，惩罚越大
CFG_SOFT_QI_WEIGHT = 0.01           # soft penalty 强度。这个值越大，惩罚越大
CFG_SOFT_QI_MAX_DELTA_SIGMA = 50.0  # 每个峰允许偏离中心的最大 sigma 数。最大范围


# ----------------- 5. 降采样策略 (Downsampling) -----------------
CFG_DOWNSAMPLE = True

# ----------------- 6. 基线校准算法 (ASLS Baseline) -----------------
CFG_ASLS_LAM = 1e4
CFG_ASLS_P_UP = 0.05
CFG_PHASE_LAM = 1e4
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
    from main_build_resonance_windows_v1 import build_resonance_windows
    from main_single_resonator_v1 import fit_single_notch_local
except ImportError as e:
    print(f"[严重错误] 缺失必要的依赖库: {e}")
    sys.exit(1)

# ==================== 配置与常量 ====================
PI_2 = 2.0 * np.pi
EPS = 1e-18

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

# ==================== 类型定义 ====================

class AslsParams(NamedTuple):
    method: str = 'asls'
    asls_lam: float = CFG_ASLS_LAM
    asls_p_upper: float = CFG_ASLS_P_UP
    asls_niter: int = 10
    phase_asls_lam: float = CFG_PHASE_LAM
    phase_asls_p: float = CFG_PHASE_P
    phase_asls_niter: int = 10

class FitOutput(TypedDict, total=False):
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
    完整 S21 模型:
        baseline * Π_i [1 - (Ql/Qc) exp(jphi) / (1 + 2jQl(f-fr)/fr)]
    """
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
    """
    返回:
        [[fr, Ql, phi, k], ...]
    """
    return np.array([[res['fr'], res['Ql'], res['phi'], res['k']] for res in results])


def _calculate_qi_from_params(fr: float, Ql: float, phi: float, k: float) -> Tuple[float, float]:
    """
    使用包含 cos(phi) 的精确公式提取 Qi 和 Qc:
        k = Ql / Qc
        1/Qi = 1/Ql - cos(phi)/Qc
    """
    if abs(k) > 1e-9:
        Qc = Ql / k
        inv_qi = 1.0 / Ql - np.cos(phi) / Qc
        if inv_qi > 1e-10:
            Qi = 1.0 / inv_qi
        else:
            Qi = 1e9
    else:
        Qc, Qi = 1e9, Ql
    return Qi, Qc


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

def _adaptive_downsample_with_windows(
    f: np.ndarray,
    S: np.ndarray,
    windows: List[Tuple[float, float]],
    indep_results: List[Dict],
    verbose: bool = True,
    background_target_points: int = 2000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    降采样策略（窗口内全保留，窗口外降采样）

    Parameters
    ----------
    f : np.ndarray
        频率数组
    S : np.ndarray
        复数 S21 数据
    windows : List[Tuple[float, float]]
        谐振窗口列表，每个元素为 (left, right)
    indep_results : List[Dict]
        独立拟合结果（此版本中不直接使用，保留参数仅为兼容原调用接口）
    verbose : bool
        是否打印降采样信息
    background_target_points : int
        希望在窗口外背景区域大约保留的点数

    Returns
    -------
    f_ds : np.ndarray
        降采样后的频率
    S_ds : np.ndarray
        降采样后的 S21
    keep_indices_arr : np.ndarray
        保留点对应的原始索引
    """
    n_points = len(f)
    keep_indices = set()

    # =====================================================
    # 1) 构造“谐振窗口内部”掩膜：窗口内全部保留
    # =====================================================
    resonance_mask = np.zeros(n_points, dtype=bool)

    for (l, r) in windows:
        mask = (f >= l) & (f <= r)
        resonance_mask |= mask

    resonance_idx = np.where(resonance_mask)[0]
    keep_indices.update(resonance_idx)

    # =====================================================
    # 2) 窗口外背景区域降采样
    # =====================================================
    background_idx = np.where(~resonance_mask)[0]

    if len(background_idx) > 0:
        step = max(1, len(background_idx) // background_target_points)
        bg_keep = background_idx[::step]
        keep_indices.update(bg_keep)
    else:
        bg_keep = np.array([], dtype=int)

    # =====================================================
    # 3) 强制保留首尾点
    # =====================================================
    keep_indices.add(0)
    keep_indices.add(n_points - 1)

    # =====================================================
    # 4) 排序并输出
    # =====================================================
    keep_indices_sorted = sorted(list(keep_indices))
    keep_indices_arr = np.array(keep_indices_sorted, dtype=int)

    if verbose:
        print(f"  降采样: {n_points} -> {len(keep_indices_arr)} "
              f"({len(keep_indices_arr) / n_points * 100:.2f}%)")
        print(f"    - 窗口内点全部保留: {len(resonance_idx)}")
        print(f"    - 窗口外点: {len(background_idx)} -> {len(bg_keep)}")
        print(f"    - 背景目标点数: {background_target_points}")

    return f[keep_indices_arr], S[keep_indices_arr], keep_indices_arr

# =========================================================================================
# ==================== 标记工具 (tools) ====================
# =========================================================================================
def _mark_overlapping_windows(windows: List[Tuple[float, float]]) -> List[bool]:
    """
    判断每个窗口是否与相邻窗口存在重叠。
    
    Parameters
    ----------
    windows : List[Tuple[float, float]]
        窗口列表，默认已按频率从低到高排列

    Returns
    -------
    overlap_flags : List[bool]
        与 windows 等长，True 表示该窗口与前一个或后一个窗口有重叠
    """
    n = len(windows)
    overlap_flags = [False] * n

    for i in range(n - 1):
        l1, r1 = windows[i]
        l2, r2 = windows[i + 1]

        # 如果相邻窗口有交叠
        if l2 < r1:
            overlap_flags[i] = True
            overlap_flags[i + 1] = True

    return overlap_flags


# =========================================================================================
# ==================== 全局优化器 (Soft shared Qi) ====================
# =========================================================================================

def _optimize_global_robust(
    f: np.ndarray,
    S_data: np.ndarray,
    independent_results: List[Dict],
    verbose: bool = True,
    weights: Optional[np.ndarray] = None,
    qi_bound_ratio: float = CFG_QI_BOUND_RATIO,
    qc_bound_ratio: float = CFG_QC_BOUND_RATIO,
    sigma_log_qi: float = CFG_SOFT_QI_SIGMA_LOG,
    soft_qi_weight: float = CFG_SOFT_QI_WEIGHT,
    max_delta_sigma: float = CFG_SOFT_QI_MAX_DELTA_SIGMA,
    **kwargs
) -> Dict:
    """
    Soft shared Qi 全局优化器

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
        fr_margin = bw * 0.02   # 保持你原来的 Ultimate 版本频率锁定策略
        phi_margin = 0.1        # 保持原来的 phi 锁定策略

        qc_min_bound = max(Qc * (1.0 - qc_bound_ratio), 1e2)
        qc_max_bound = max(Qc * (1.0 + qc_bound_ratio), 1e3)

        bounds_low.extend([
            fr - fr_margin,
            qc_min_bound,
            phi - phi_margin,
            dlog_lo
        ])
        bounds_high.extend([
            fr + fr_margin,
            qc_max_bound,
            phi + phi_margin,
            dlog_hi
        ])

    x0_real = np.array(p_global + p_local, dtype=float)
    bounds_low_real = np.array(bounds_low, dtype=float)
    bounds_high_real = np.array(bounds_high, dtype=float)
    scales_arr = np.array(scales, dtype=float)

    # 将初值压到 bounds 内部，避免 x0 越界
    x0_real = _clip_inside_bounds(x0_real, bounds_low_real, bounds_high_real)

    # ------------------------------------------------------------
    # 4. 参数归一化
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

        fr_arr = local_p[0::4]
        Qc_arr = local_p[1::4]
        phi_arr = local_p[2::4]
        dlog_arr = local_p[3::4]

        log_qi_arr = log_qi_center + dlog_arr
        Qi_arr = np.exp(log_qi_arr)

        safe_Qc = np.maximum(Qc_arr, 1e-12)
        inv_qi = 1.0 / Qi_arr
        inv_qc_real = np.cos(phi_arr) / safe_Qc
        inv_ql = np.maximum(inv_qi + inv_qc_real, 1e-10)
        Ql_arr = 1.0 / inv_ql

        model_res_params = np.column_stack((fr_arr, Ql_arr, Qc_arr, phi_arr)).ravel()

        S_model = s21_model_total(f, model_res_params, base_p, n_res)
        resid_complex = (S_data - S_model)

        resid_data = np.hstack([resid_complex.real, resid_complex.imag]) * np.hstack([w_sqrt, w_sqrt])

        # soft shared Qi penalty
        soft_qi_resid = soft_qi_weight * (dlog_arr / sigma_log_qi)

        return np.hstack([resid_data, soft_qi_resid])

    if verbose:
        print(f"  [Soft Shared Qi Fit] 启动归一化优化 (Linear Loss)...")
        print(f"  -> 初始 Qi_center: {qi_center_init:.0f}")
        print(f"  -> Qi_center 允许范围: {qi_min:.0f} ~ {qi_max:.0f}")
        print(f"  -> Soft Qi sigma(log): {sigma_log_qi:.3f}, weight: {soft_qi_weight:.3f}")
        print(f"  -> 每个峰 dlogQi 范围: [{dlog_lo:.3f}, {dlog_hi:.3f}]")

    res = least_squares(
        fun,
        x0_norm,
        bounds=bounds_norm,
        jac='2-point',
        loss='linear',
        x_scale='jac',
        xtol=1e-5,
        ftol=1e-5,
        gtol=1e-5,
        max_nfev=200,
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

    S_fit = s21_model_total(f, np.array(final_res_p), final_base, n_res)

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

    final_resid_abs2 = np.abs(S_data - S_fit) ** 2
    eval_weights = weights if weights is not None else np.ones_like(S_data)
    weighted_residual = np.sqrt(np.average(final_resid_abs2, weights=eval_weights))

    return {
        'resonators': resonators,
        'S_fit': S_fit,
        'baseline_params': final_base,
        'qi_center_soft': final_qi_center,
        'residual': weighted_residual,
        'scheme_name': 'Global_Soft_SharedQi_Reparam_Normalized'
    }

# =========================================================================================
# ==================== 主拟合流程 ====================
# =========================================================================================

def hierarchical_fit_s21_notch_multi(
    f: np.ndarray,
    S: np.ndarray,
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

    if params is None:
        params = AslsParams()

    if verbose:
        print("=" * 60)
        print("开始 S21 多谐振器全局拟合 (Ultimate Edition, Soft shared Qi)")
        print("=" * 60)

    # 1. 预处理
    if verbose:
        print("\n>>> Step 1: 信号校准 (粗调)...")
    tau0 = estimate_tau_robust(f, S, plot=tau_plot)
    S_tau = S * np.exp(1j * PI_2 * f * tau0)

    f_sort, S_flat, A_hat, base_db, base_ph = amplitude_baseline_normalize(
        f, S_tau, params=params, plot=preview_plot
    )

    # 2. 检测谐振窗口
    if verbose:
        print("\n>>> Step 2: 谐振窗口检测...")
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

    window_overlap_flags = _mark_overlapping_windows(windows)

    # 3. 独立拟合
    if verbose:
        print("\n>>> Step 3: 独立拟合 (每个 resolved window 直接拟合)...")
    
    indep_results = []
    window_info = []
    
    for i, (l, r) in enumerate(windows):
        mask = (f_sort >= l) & (f_sort <= r)
        f_win, S_win = f_sort[mask], S_flat[mask]
    
        # 最少点数检查
        if len(f_win) < max(5, min_pts_per_win):
            window_info.append({
                'window': (l, r),
                'status': 'too_few_points',
                'n_points': len(f_win)
            })
            continue
    
        try:
            fit_res = fit_single_notch_local(f_win, S_win, plot=False)
    
            # 基本合法性检查
            if not (f_win[0] <= fit_res['fr'] <= f_win[-1] and 0 <= fit_res['Ql'] <= 1e9):
                raise ValueError("Fit out of bounds")
    
            indep_results.append(fit_res)
            window_info.append({
                'window': (l, r),
                'status': 'ok',
                'n_points': len(f_win)
            })
    
            if verbose:
                fr_val = fit_res['fr']
                Ql_val = fit_res['Ql']
                phi_val = fit_res['phi']
                k_val = fit_res['k']
                Qi_val, Qc_val = _calculate_qi_from_params(fr_val, Ql_val, phi_val, k_val)
                
                
                overlap_tag = " [OVERLAP]" if window_overlap_flags[i] else ""
                
                print(
                    f"    [初值 {len(indep_results)}{overlap_tag}] "
                    f"fr: {fr_val/1e9:.6f} GHz, "
                    f"Ql: {Ql_val:.0f}, "
                    f"Qc: {Qc_val:.0f}, "
                    f"Qi: {Qi_val:.0f}, "
                    f"phi: {phi_val:.3f}, "
                    f"pts: {len(f_win)}"
                )

    
        except Exception as e:
            window_info.append({
                'window': (l, r),
                'status': f'fit_failed: {e}',
                'n_points': len(f_win)
            })
            if verbose:
                print(f"    [初值] 窗口 {i+1} 拟合失败: {e}")
    
    if not indep_results:
        raise RuntimeError("无法提取任何谐振器初值！")
    
    if verbose:
        print(f"Step 3 完成: 提取 {len(indep_results)} 个有效谐振峰")


    # 4. 降采样
    f_opt, S_opt = f_sort, S_flat
    ds_info = None
    if enable_downsample:
        f_opt, S_opt, _ = _adaptive_downsample_with_windows(f_sort, S_flat, windows, indep_results, verbose)
        ds_info = {'orig': len(f_sort), 'new': len(f_opt)}

    # 5. 全局优化
    if verbose:
        print("\n>>> Step 4: 全局联合优化...")

    # 已修改：将基线的权重极度压低，让峰值区域绝对主导
    weights = np.ones_like(f_opt) * 0.01
    if use_weighting:
        for res in indep_results:
            fr_val, Ql_val = res['fr'], res['Ql']
            sigma = fr_val / Ql_val * weighting_sigma_factor
            w = np.exp(-0.5 * ((f_opt - fr_val) / sigma) ** 2)
            weights = np.maximum(weights, w)

    g_res = _optimize_global_robust(
        f_opt,
        S_opt,
        indep_results,
        verbose=verbose,
        weights=weights,
        qi_bound_ratio=CFG_QI_BOUND_RATIO,
        qc_bound_ratio=CFG_QC_BOUND_RATIO,
        sigma_log_qi=CFG_SOFT_QI_SIGMA_LOG,
        soft_qi_weight=CFG_SOFT_QI_WEIGHT,
        max_delta_sigma=CFG_SOFT_QI_MAX_DELTA_SIGMA
    )

    final_results = g_res['resonators']
    final_base_p = g_res['baseline_params']

    # 6. 重建结果
    if verbose:
        print("\n>>> Step 5: 重建最终结果...")

    final_res_p = []
    for res_item in final_results:
        final_res_p.extend([res_item['fr'], res_item['Ql'], res_item['Qc'], res_item['phi']])

    S_fit_cal_full = s21_model_total(f_sort, np.array(final_res_p), final_base_p, len(final_results))
    S_fit_final = S_fit_cal_full * A_hat * np.exp(1j * base_ph) * np.exp(-1j * PI_2 * f_sort * tau0)

    output: FitOutput = {
        'global_params': {
            'tau': tau0,
            'f_range': [f_sort.min(), f_sort.max()],
            'baseline_params': final_base_p,
            'qi_center_soft': g_res.get('qi_center_soft', None)
        },
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

# =========================================================================================
# ==================== 绘图函数 ====================
# =========================================================================================

def plot_fit_results(f: np.ndarray, S: np.ndarray, result: Dict, save_path: Optional[str] = None):
    import numpy as np
    import matplotlib.pyplot as plt

    S_fit = result['S21_fit']
    f_GHz = f / 1e9

    # =========================
    # 统一配色
    # =========================
    data_color = '#0000FF'        # 蓝：原始数据
    fit_color = '#FFA500'         # 橙：拟合结果
    mag_err_color = '#00FFFF'     # 青：幅度误差
    phase_err_color = '#FF00FF'   # 洋红：相位误差
    guide_color = '#666666'       # 灰：辅助线

    # =========================
    # 辅助函数：unwrap + 去线性趋势
    # =========================
    def unwrap_and_detrend_phase(freq, S_complex):
        """
        对相位进行：
        1) unwrap
        2) 一次多项式拟合并去除线性趋势
        返回：
            phase_detrended, phase_unwrapped, trend
        """
        phase_unwrapped = np.unwrap(np.angle(S_complex))
        p = np.polyfit(freq, phase_unwrapped, 1)
        trend = np.polyval(p, freq)
        phase_detrended = phase_unwrapped - trend
        return phase_detrended, phase_unwrapped, trend

    # =========================
    # 普通误差（非加权）
    # =========================
    mag_data = np.abs(S)
    mag_fit = np.abs(S_fit)
    mag_err = mag_data - mag_fit

    # 相位：先 unwrap 再去线性趋势
    phase_dt, phase_unwrap, trend_data = unwrap_and_detrend_phase(f, S)
    phase_fit_dt, phase_fit_unwrap, trend_fit = unwrap_and_detrend_phase(f, S_fit)

    # 相位误差：用 detrended unwrapped phase 的差值
    phase_err = phase_dt - phase_fit_dt

    # RMS（普通 RMS，不加权）
    mag_rms = np.sqrt(np.mean(mag_err ** 2))
    phase_rms = np.sqrt(np.mean(phase_err ** 2))

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

    # =========================
    # 创建图像：4行1列
    # =========================
    fig, axes = plt.subplots(4, 1, figsize=(11, 13), sharex=True)

    # ======================================================
    # (a) 幅度拟合
    # ======================================================
    ax = axes[0]
    ax.plot(
        f_GHz, 20 * np.log10(np.abs(S)),
        color=data_color, ls='-', lw=2.4, label='Raw amplitude'
    )
    ax.plot(
        f_GHz, 20 * np.log10(np.abs(S_fit)),
        color=fit_color, ls='--', lw=2.6, label='Global fit'
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
        color=data_color, ls='-', lw=2.4, label='Raw detrended phase'
    )
    ax.plot(
        f_GHz, phase_fit_dt,
        color=fit_color, ls='--', lw=2.6, label='Global fit detrended phase'
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
        color=mag_err_color, ls='-', lw=2.0, label='Magnitude error'
    )
    ax.axhline(0, color=guide_color, lw=0.8, ls=':', alpha=0.7)
    ax.set_ylabel(r'$|S_{21}|$ error')
    ax.set_title(f'(c) $|S_{{21}}|$ Error (RMS = {mag_rms:.2e})')
    ax.grid(True, alpha=0.3, ls='--', lw=0.6)
    ax.tick_params(direction='in')

    # ======================================================
    # (d) 相位误差（基于 detrended unwrapped phase）
    # ======================================================
    ax = axes[3]
    ax.plot(
        f_GHz, phase_err,
        color=phase_err_color, ls='-', lw=2.0, label='Phase error'
    )
    ax.axhline(0, color=guide_color, lw=0.8, ls=':', alpha=0.7)
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('Phase error (rad)')
    ax.set_title(f'(d) Phase Error (Detrended, RMS = {phase_rms:.2e})')
    ax.grid(True, alpha=0.3, ls='--', lw=0.6)
    ax.tick_params(direction='in')

    # =========================
    # 布局优化
    # =========================
    plt.subplots_adjust(
        left=0.11, right=0.97, top=0.96, bottom=0.08,
        hspace=0.28
    )

    if save_path:
        fig.savefig(save_path, dpi=600, bbox_inches='tight')

    plt.show()

# =========================================================================================
# ==================== 主入口 ====================
# =========================================================================================

def main():
    import time
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=str, nargs='?', default=DEFAULT_FILE)
    parser.add_argument("--no-plot", action="store_true", help="禁用绘图")
    args = parser.parse_args()

    if not os.path.exists(args.file_path):
        print(f"File not found: {args.file_path}")
        return

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

        print("\n" + "=" * 84)
        print("Final Fitting Results (Ultimate Edition, Soft shared Qi)")
        print("=" * 84)
        print(f"{'Idx':<4} {'Freq(GHz)':<12} {'Ql':<10} {'Qi':<10} {'Qc':<10} {'Phi(rad)':<10} {'dlogQi':<10}")
        print("-" * 84)

        results_list = []
        for i, r in enumerate(res['resonators']):
            print(
                f"{i+1:<4} "
                f"{r['fr']/1e9:<12.6f} "
                f"{r['Ql']:<10.0f} "
                f"{r['Qi']:<10.0f} "
                f"{r['Qc']:<10.0f} "
                f"{r['phi']:<10.5f} "
                f"{r.get('dlogQi', 0.0):<10.4f}"
            )

            results_list.append([
                i + 1,
                r['fr'] / 1e9,
                r['Ql'],
                r['Qi'],
                r['Qc'],
                r['phi'],
                r['k'],
                r.get('dlogQi', 0.0)
            ])

        if 'qi_center_soft' in res['global_params'] and res['global_params']['qi_center_soft'] is not None:
            print("\n=== Soft Shared Qi Center ===")
            print(f"Qi_center_soft = {res['global_params']['qi_center_soft']:.0f}")

        if 'baseline_params' in res['global_params']:
            bp = res['global_params']['baseline_params']
            print("\n=== Final Baseline Params ===")
            print(f"A            = {bp[0]:.6f}")
            print(f"alpha        = {bp[1]:.6f}")
            print(f"tau_residual = {bp[2]:.6e}")

        save_dir = os.path.dirname(args.file_path)
        csv_name = f"fit_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        csv_path = os.path.join(save_dir, csv_name)

        df = pd.DataFrame(
            results_list,
            columns=['Index', 'Freq_GHz', 'Ql', 'Qi', 'Qc', 'Phi', 'k', 'dlogQi']
        )
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    print(f"\nTotal Time: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()