# -*- coding: utf-8 -*-
"""
MKID/超导谐振器 S21 全局拟合算法 (最终增强版)
包含: 动态基线优化 + Soft Qi 约束 + Huber Loss 抗噪
Created on: 2026-03-02
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import warnings
import argparse
from typing import Dict, List, Optional, NamedTuple, Tuple, Any

try:
    from typing import TypedDict
except ImportError:
    from typing import Dict as TypedDict

from scipy.optimize import least_squares
from itertools import combinations

# =========================================================================================
#                                【 用户可调参数配置区 】
# =========================================================================================

# ----------------- 1. 文件与基础配置 -----------------
DEFAULT_FILE = r'S21_simulation.txt'   # [默认文件]: 默认读取的 S21 数据文件路径 (.s2p 或 .txt)

# ----------------- 2. 谐振峰检测 (Peak Detection) -----------------
# 物理意义：控制哪些"坑"会被识别为真实的超导谐振峰，避免将基线噪声或驻波误认为谐振器。
CFG_DISTANCE = 30          # [最小点距] (点数): 两个相邻被识别峰之间的最小数据点数。用于防止在一个宽峰上检测到多个伪峰。
CFG_MIN_Q = 1000.0         # [最小 Q 值]: 物理品质因数下限。若粗估出的 Q 值低于此值，则认为是宽频背景波动而非超导谐振器。

# ----------------- 3. 优化算法控制 (Optimization Control) -----------------
# 物理意义：控制全局 Least Squares 拟合时的策略、正则化约束和权重分布。
CFG_USE_WEIGHT = True      # [是否启用权重]: 是否在拟合时对中心峰值附近赋予更高的权重。推荐 True，能让中心拟合得更准。
CFG_WEIGHT_SIGMA = 3.0     # [权重分布宽度]: 高斯权重的覆盖宽度(线宽的倍数)。决定在峰中心外多远距离权重开始衰减。
CFG_QI_BOUND_RATIO = 0.5   # [Qi 波动范围]: 0.3 表示允许全局共享的 Qi 在初始估算值的 ±30% 范围内自由寻找最优解。

# ----------------- 4. 降采样策略 (Downsampling) -----------------
# 物理意义：保留关键信息(峰周围)同时大幅度稀疏化非谐振区域(基线)，成百倍提升拟合速度。
CFG_DOWNSAMPLE = True      # [是否启用降采样]: 推荐开启，关闭后大数据量下拟合极慢。

# ----------------- 5. 基线校准算法 (ASLS Baseline) -----------------
# 物理意义：处理非对称最小二乘平滑滤波，用于扣除由放大器、电缆引起的背景波动。
CFG_ASLS_LAM = 1e6         # [幅度基线平滑度]: 惩罚因子，数值越大估算出的背景基线越平滑(僵硬)；越小越容易顺应局部坑洼。
CFG_ASLS_P_UP = 0.05       # [幅度基线非对称权重]: 较小的值表示让基线贴合数据的上沿(因为谐振通常是往下的坑)。
CFG_PHASE_LAM = 1e6        # [相位基线平滑度]: 相位基线的僵硬程度。
CFG_PHASE_P = 0.5          # [相位基线非对称权重]: 0.5 表示正负偏差一视同仁(走中间)。

# ----------------- 6. 可视化与调试 (Plot & Debug) -----------------
CFG_PLOT_FINAL = True      # [绘制最终结果]: 是否弹出最终的 4 个子图 (幅度、相位、IQ、残差)。
CFG_VERBOSE = True         # [输出详细日志]: 终端是否输出拟合的进度和详细日志。

# =========================================================================================

# ==================== 依赖检查 ====================
try:
    from main_data_reader import read_s2p, read_txt
    from main_estimate_delay import estimate_tau_robust
    from main_amplitude_baseline_normalize import amplitude_baseline_normalize
    from main_build_resonance_windows import build_resonance_windows
    from main_single_resonator import fit_single_notch_local
except ImportError as e:
    print(f"[严重错误] 缺失库: {e}")
    sys.exit(1)

# ==================== 配置 ====================
PI_2 = 2.0 * np.pi

def configure_plotting():
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    plt.rcParams['font.family'] = 'serif'
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

class FitOutput(TypedDict):
    global_params: Dict[str, Any]
    resonators: List[Dict]
    window_info: List[Dict]
    S21_fit: np.ndarray
    optimization_scheme: str
    downsample_info: Optional[Dict]

# ==================== 核心物理模型 (已升级) ====================

def s21_model_total(f: np.ndarray, res_params_flat: np.ndarray, baseline_params: np.ndarray, n_res: int) -> np.ndarray:
    """
    包含动态基线和延时的完整 S21 模型 (向量化)
    res_params_flat: [fr1, Ql1, Qc1, phi1, fr2, ...] (注意: 这里传入的是 Qc 而不是 k)
    baseline_params: [A, alpha, tau_residual]
    """
    # 1. 解析基线参数
    A, alpha, tau = baseline_params
    
    # 2. 计算基线 (Baseline + Delay)
    # 这里的 f 是频率 Hz
    S_baseline = A * np.exp(1j * (alpha + PI_2 * f * tau))
    
    # 3. 计算谐振器 (乘积形式)
    # reshape: (n_res, 4) -> [fr, Ql, Qc, phi]
    p = res_params_flat.reshape(n_res, 4)
    fr = p[:, 0]
    Ql = p[:, 1]
    Qc = p[:, 2]
    phi = p[:, 3]
    
    # 这里的 k 是由 Qc 算出来的
    # 防止 Qc 为 0
    # Ql/Qc
    k = np.zeros_like(Ql)
    mask = np.abs(Qc) > 1e-9
    k[mask] = Ql[mask] / Qc[mask]
    
    # x = (f - fr) / fr
    x = (f[:, None] - fr) / fr 
    
    denom = 1.0 + 2j * Ql * x
    # S21 = 1 - (Ql/Qc * e^(j*phi)) / (1 + 2j*Ql*x)
    term = k * np.exp(1j * phi) / denom
    
    S_res = np.prod(1.0 - term, axis=1)
    
    return S_res * S_baseline

def _extract_resonator_arrays(results: List[Dict]) -> np.ndarray:
    return np.array([[res['fr'], res['Ql'], res['phi'], res['k']] for res in results])

def _calculate_qi_from_params(fr, Ql, phi, k):
    if abs(k) > 1e-9:
        Qc = Ql / k
        inv_qi = 1.0/Ql - 1.0/Qc
        if inv_qi > 1e-10: 
            Qi = 1.0 / inv_qi
        else:
            Qi = 1e8 # Limit
    else:
        Qc, Qi = 1e9, Ql
    return Qi, Qc

# ==================== 降采样逻辑 ====================

def _adaptive_downsample_with_windows(f, S, windows, indep_results, verbose=True):
    # (保持原有的优秀逻辑不变，这里简化显示)
    n_points = len(f)
    keep_indices = set()
    
    for i, (l, r) in enumerate(windows):
        idx = np.where((f >= l) & (f <= r))[0]
        if len(idx) == 0: continue
        
        # 简单策略：窗口内保留
        step = max(1, len(idx) // 200) # 限制每个窗口点数
        keep_indices.update(idx[::step])
        
    # 加上一些基线点
    all_idx = np.arange(0, n_points, max(1, n_points//500))
    keep_indices.update(all_idx)
    
    keep_indices = sorted(list(keep_indices))
    if verbose: print(f"  降采样: {n_points} -> {len(keep_indices)}")
    
    return f[keep_indices], S[keep_indices], np.array(keep_indices)

# ==================== 全局优化器 (核心升级) ====================

def _optimize_global_robust(
    f: np.ndarray, S_data: np.ndarray, 
    independent_results: List[Dict],
    verbose: bool = True,
    weights: Optional[np.ndarray] = None,
    qi_bound_ratio: float = 0.3  # 允许 Qi 偏离初始中位数的比例
) -> Dict:
    """
    [终极修复版] 参数完全归一化 + 共享 Qi
    彻底解决因量级悬殊(1e-9 到 1e9)导致的梯度爆炸和优化器早停问题。
    """
    n_res = len(independent_results)
    res_arrays = _extract_resonator_arrays(independent_results) 
    
    # 1. 估算初始 Shared Qi
    qi_vals = []
    for row in res_arrays:
        qi, _ = _calculate_qi_from_params(*row)
        if 0 < qi < 1e9: qi_vals.append(qi)
    qi_shared_init = np.median(qi_vals) if qi_vals else 1e5
    
    p_global = [1.0, 0.0, 0.0, qi_shared_init] 
    
    # ==========================================
    # 【核心修复1】建立物理缩放因子 (Scales)
    # 保证所有的因子都绝对大于0，防止边界反转
    # ==========================================
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
        # phi 可能会等于 0，所以给它一个底线 1.0
        scales.extend([fr, Qc, max(abs(phi), 1.0)])
        
        bw = fr / Ql
        fr_margin = bw * 0.002  # 锁定 fr 仅允许 2% 线宽浮动
        
        bounds_low.extend([fr - fr_margin, Qc * 0.1, phi - np.pi])
        bounds_high.extend([fr + fr_margin, Qc * 10.0, phi + np.pi])
        
    x0_real = np.array(p_global + p_local)
    bounds_low_real = np.array(bounds_low)
    bounds_high_real = np.array(bounds_high)
    scales_arr = np.array(scales)
    
    # ==========================================
    # 【核心修复2】执行彻底的归一化
    # 优化器看到的 x0 将全是 1.0 附近的值！
    # ==========================================
    x0_norm = x0_real / scales_arr
    bounds_norm = (bounds_low_real / scales_arr, bounds_high_real / scales_arr)
    
    w_sqrt = np.sqrt(weights) if weights is not None else 1.0

    def fun(p_norm):
        # 计算时：将归一化的参数还原为真实的物理量
        p_real = p_norm * scales_arr
        
        base_p = p_real[:3]      
        qi_shared = p_real[3]    
        local_p = p_real[4:]     
        
        model_res_params = []
        for i in range(n_res):
            fr, Qc, phi = local_p[i*3 : (i+1)*3]
            
            # 【核心物理修正】引入 cos(phi) 投影
            # 1/Ql = 1/Qi + cos(phi)/Qc => Ql = 1 / (1/Qi + cos(phi)/Qc)
            inv_qi = 1.0 / qi_shared
            inv_qc_real = np.cos(phi) / Qc if Qc > 0 else 0.0
            
            # 防止极少数情况下 phi>90度导致分母为负
            inv_ql = max(inv_qi + inv_qc_real, 1e-10) 
            Ql = 1.0 / inv_ql
            
            model_res_params.extend([fr, Ql, Qc, phi])
            
        S_model = s21_model_total(f, np.array(model_res_params), base_p, n_res)
        resid_complex = (S_data - S_model)
        return np.hstack([resid_complex.real, resid_complex.imag]) * np.hstack([w_sqrt, w_sqrt])

    if verbose: 
        print(f"  [Shared Qi Fit] 启动归一化优化...")
        print(f"  -> 初始 Shared Qi: {qi_shared_init:.0f} (允许波动范围: {qi_min:.0f} ~ {qi_max:.0f})")
    
    # 恢复为 linear loss，因为归一化后梯度极其灵敏，不需要抗噪函数压制梯度了
    res = least_squares(
        fun, x0_norm, bounds=bounds_norm, 
        jac='3-point', 
        loss='linear',          # <== 改回 linear，保证微小梯度也能生效
        x_scale='jac', 
        xtol=1e-12, ftol=1e-12, gtol=1e-12,
        max_nfev=300, 
        verbose=2 if verbose else 0
    )
    
    # ==========================================
    # 【核心修复3】将优化完的结果还原
    # ==========================================
    final_real = res.x * scales_arr
    
    final_base = final_real[:3]
    final_qi = final_real[3] 
    final_local = final_real[4:]
    
    resonators = []
    final_res_p = []
    
    for i in range(n_res):
        fr, Qc, phi = final_local[i*3 : (i+1)*3]
        
        # ==========================================
        # 【同步修改】使用带 cos(phi) 的严格物理公式反算最终 Ql
        # ==========================================
        inv_qi = 1.0 / final_qi
        inv_qc_real = np.cos(phi) / Qc if Qc > 0 else 0.0
        
        # 防止极端情况分母为 0 或负数
        inv_ql = max(inv_qi + inv_qc_real, 1e-10)
        Ql = 1.0 / inv_ql
        
        # 计算阻抗不对称度 k (复数 k 的模，用于给外部调用时兼容)
        k = Ql / Qc if Qc > 0 else 0.0
        
        resonators.append({'fr': fr, 'Ql': Ql, 'Qc': Qc, 'Qi': final_qi, 'phi': phi, 'k': k})
        final_res_p.extend([fr, Ql, Qc, phi])
        
    S_fit_final = s21_model_total(f, np.array(final_res_p), final_base, n_res)
    
    if verbose:
        print(f"  -> 优化完成! 最终 Shared Qi: {final_qi:.0f} (改变了 {(final_qi - qi_shared_init)/qi_shared_init*100:.2f}%)")
    
    return {
        'resonators': resonators,
        'S_fit': S_fit_final,
        'baseline_params': final_base,
        'residual': np.sqrt(np.mean(np.abs(S_data - S_fit_final)**2)),
        'scheme_name': 'Robust_Global_Shared_Qi_Normalized'
    }

# ==================== 主流程 (已更新) ====================

def hierarchical_fit_s21_notch_multi(
    f: np.ndarray, S: np.ndarray, 
    params: Optional[AslsParams] = None,
    enable_downsample: bool = CFG_DOWNSAMPLE,
    use_weighting: bool = CFG_USE_WEIGHT,
    verbose: bool = CFG_VERBOSE,
    plot_result: bool = CFG_PLOT_FINAL,
    save_path: Optional[str] = None
) -> FitOutput:
    
    # [新增] 如果 params 为 None，自动创建一个默认对象
    if params is None:
        params = AslsParams()
    
    # 1. 预处理 (粗基线)
    if verbose: print(">>> Step 1: 粗基线去除...")
    tau0 = estimate_tau_robust(f, S, plot=False)
    S_tau = S * np.exp(1j * PI_2 * f * tau0)
    f_sort, S_flat, A_hat, base_db, base_ph = amplitude_baseline_normalize(f, S_tau, params=params)
    
    # 2. 找峰
    if verbose: print(">>> Step 2: 谐振峰检测...")
    windows = build_resonance_windows(f_sort, S_flat, min_Q=CFG_MIN_Q, distance=CFG_DISTANCE)
    if verbose: print(f"    检测到 {len(windows)} 个谐振模式")
    
        # 3. 独立拟合 (初值)
    if verbose: print(">>> Step 3: 独立拟合 (初值)...")
    indep_results = []
    win_info = []
    for i, (l, r) in enumerate(windows):
        mask = (f_sort >= l) & (f_sort <= r)
        try:
            res = fit_single_notch_local(f_sort[mask], S_flat[mask])
            indep_results.append(res)
            win_info.append({'window': (l,r), 'status': 'ok'})
            
            # 打印独立拟合得到的初值
            if verbose:
                fr_val = res['fr']
                Ql_val = res['Ql']
                phi_val = res['phi']
                k_val = res['k']
                Qi_val, Qc_val = _calculate_qi_from_params(fr_val, Ql_val, phi_val, k_val)
                print(f"    [初值 {len(indep_results)}] fr: {fr_val/1e9:.6f} GHz, Ql: {Ql_val:.0f}, Qc: {Qc_val:.0f}, Qi: {Qi_val:.0f}, phi: {phi_val:.3f}")
                
        except Exception as e:
            if verbose:
                print(f"    [初值] 窗口 {i+1} 拟合失败: {e}")
            pass
            
    if not indep_results: raise RuntimeError("无有效谐振器")

    # 4. 准备全局数据
    f_opt, S_opt = f_sort, S_flat
    ds_info = None
    if enable_downsample:
        f_opt, S_opt, idxs = _adaptive_downsample_with_windows(
            f_sort, S_flat, windows, indep_results, verbose
        )
        ds_info = {'orig': len(f_sort), 'new': len(f_opt)}
        
    # 5. 全局优化 (Robust Mode)
    if verbose: print(">>> Step 4: 全局优化 (Robust Mode)...")
    
    # 简单的权重 (谐振峰区域加权)
    weights = np.ones_like(f_opt) * 0.1
    if use_weighting:
        for r in indep_results:
            w = np.exp(-0.5 * ((f_opt - r['fr'])/(r['fr']/r['Ql']*CFG_WEIGHT_SIGMA))**2)
            weights = np.maximum(weights, w)


    g_res = _optimize_global_robust(
        f_opt, S_opt, indep_results, 
        verbose=verbose, 
        weights=weights, 
        qi_bound_ratio=CFG_QI_BOUND_RATIO  # <== 【关键修改】改为传入波动范围
    )
    
    # 6. 重建最终结果 (还原到原始数据尺度)
    # S_final = S_fit_optimized * (粗基线增益) * (粗延时)
    # 注意: g_res['S_fit'] 仅对应 f_opt，我们需要重新计算全谱
    
    final_res_p = []
    for r in g_res['resonators']:
        final_res_p.extend([r['fr'], r['Ql'], r['Qc'], r['phi']])
    final_base_p = g_res['baseline_params']
    
    # 计算全谱 S_fit (包含微调基线)
    S_fit_flat_full = s21_model_total(f_sort, np.array(final_res_p), final_base_p, len(indep_results))
    
    # 加上之前的粗基线和延时
    S_final = S_fit_flat_full * A_hat * np.exp(1j * base_ph) * np.exp(-1j * PI_2 * f_sort * tau0)
    
    output = {
        'resonators': g_res['resonators'],
        'S21_fit': S_final,
        'global_params': {'tau': tau0, 'base_db': base_db},
        'window_info': win_info,
        'optimization_scheme': 'Robust_Global',
        'downsample_info': ds_info
    }
    
    if plot_result:
        plot_fit_results(f_sort, S, output, save_path)
        
    return output

# ==================== 绘图与入口 ====================

def plot_fit_results(f, S, result, save_path=None):
    S_fit = result['S21_fit']
    resid = np.abs(S - S_fit)
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    ax = axes[0,0]
    ax.plot(f/1e9, 20*np.log10(np.abs(S)), 'b.', alpha=0.3, label='Data')
    ax.plot(f/1e9, 20*np.log10(np.abs(S_fit)), 'r-', lw=1, label='Fit')
    ax.set_ylabel('|S21| (dB)'); ax.legend()
    
    ax = axes[0,1]
    ax.plot(f/1e9, np.angle(S), 'b.', alpha=0.3)
    ax.plot(f/1e9, np.angle(S_fit), 'r-', lw=1)
    ax.set_ylabel('Phase (rad)')
    
    ax = axes[1,0] # IQ
    ax.plot(S.real, S.imag, 'b.', alpha=0.3)
    ax.plot(S_fit.real, S_fit.imag, 'r-', lw=1)
    ax.axis('equal'); ax.set_xlabel('I'); ax.set_ylabel('Q')

    ax = axes[1,1] # Residual
    ax.plot(f/1e9, resid, 'k-', lw=1)
    ax.set_ylabel('Residual Abs'); ax.set_title(f"RMS: {np.sqrt(np.mean(resid**2)):.2e}")
    
    plt.tight_layout()
    if save_path: plt.savefig(save_path)
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=str, nargs='?', default=DEFAULT_FILE)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    
    if not os.path.exists(args.file_path):
        print(f"File not found: {args.file_path}"); return

    try:
        # 读取数据
        if args.file_path.endswith('.s2p'):
            f, S = read_s2p(args.file_path)
        else:
            f, S = read_txt(args.file_path)
            
        print(f"Data Loaded: {len(f)} pts")
        
        # ---------------------------------------------------------
        # [关键修复] 这里需要初始化 params 对象！
        # ---------------------------------------------------------
        fit_params = AslsParams() 
        
        # 将 fit_params 传入函数
        do_plot = CFG_PLOT_FINAL if not args.no_plot else False
        res = hierarchical_fit_s21_notch_multi(
            f, S, 
            params=fit_params,  
            enable_downsample=CFG_DOWNSAMPLE, 
            plot_result=do_plot
        )
        
        # 打印包含 phi 的最终结果
        print("\n=== Final Results ===")
        print(f"{'Idx':<4} {'Fr(GHz)':<10} {'Qi':<10} {'Qc':<10} {'Ql':<10} {'Phi(rad)':<8}")
        for i, r in enumerate(res['resonators']):
            print(f"{i+1:<4} {r['fr']/1e9:<10.6f} {r['Qi']:<10.0f} {r['Qc']:<10.0f} {r['Ql']:<10.0f} {r['phi']:<8.3f}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()