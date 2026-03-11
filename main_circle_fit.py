# circle_fit.py
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from scipy.optimize import least_squares

# 添加常量定义
EPS_MIN = 1e-12

def _circle_fit_kasa(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, np.ndarray]:
    """
    Kåsa 圆拟合方法（fallback）。
    返回: (xc, yc, r, residuals)
    """
    z2 = x * x + y * y
    A = np.column_stack([x, y, np.ones_like(x)])
    sol, *_ = np.linalg.lstsq(A, z2, rcond=None)
    xc = sol[0] / 2.0
    yc = sol[1] / 2.0
    r = np.sqrt(max(sol[2] + xc * xc + yc * yc, 0.0))
    ri = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)
    return float(xc), float(yc), float(r), (ri - r)

def circle_fit_pratt(z: np.ndarray, refine: bool = True, plot: bool = False) -> Tuple[float, float, float, np.ndarray]:
    """
    Pratt 圆拟合方法。
    
    参数:
        z: 复数数组，表示数据点坐标
        refine: 是否进行精化拟合
        plot: 是否绘制拟合结果
    
    返回: (xc, yc, r, residuals)
    """
    # 优化：避免不必要的类型转换
    x = np.real(z).astype(float, copy=False)
    y = np.imag(z).astype(float, copy=False)
    z2 = x * x + y * y
    M = np.array([
        [np.sum(z2 * z2), np.sum(x * z2), np.sum(y * z2), np.sum(z2)],
        [np.sum(x * z2),  np.sum(x * x), np.sum(x * y),  np.sum(x)],
        [np.sum(y * z2),  np.sum(x * y), np.sum(y * y),  np.sum(y)],
        [np.sum(z2),      np.sum(x),     np.sum(y),      len(x)]
    ], dtype=float)
    B = np.array([
        [0.0, 0.0, 0.0, -2.0],
        [0.0, 1.0, 0.0,  0.0],
        [0.0, 0.0, 1.0,  0.0],
        [-2.0,0.0, 0.0,  0.0]
    ], dtype=float)
    try:
        eigvals, eigvecs = np.linalg.eig(np.linalg.inv(B) @ M)
        valid = np.isfinite(eigvals) & (eigvals > 0)
        if not np.any(valid):
            raise np.linalg.LinAlgError("No positive eigenvalue found.")
        idx = np.argmin(eigvals[valid])
        v = eigvecs[:, np.where(valid)[0][idx]].real
    except (np.linalg.LinAlgError, ValueError):
        # fallback: Kåsa
        return _circle_fit_kasa(x, y)

    Acoef, Bcoef, Ccoef, Dcoef = v
    if Acoef == 0:
        return _circle_fit_kasa(x, y)

    xc = -Bcoef / (2.0 * Acoef)
    yc = -Ccoef / (2.0 * Acoef)
    r_sq = (Bcoef * Bcoef + Ccoef * Ccoef - 4.0 * Acoef * Dcoef) / (4.0 * Acoef * Acoef)
    r = np.sqrt(abs(r_sq))

    def residuals_geo(p):
        xc_, yc_, r_ = p
        ri_ = np.sqrt((x - xc_) ** 2 + (y - yc_) ** 2)
        return ri_ - r_

    if refine:
        p0 = np.array([xc, yc, r])
        res1 = least_squares(residuals_geo, p0, method="trf")
        xc, yc, r = res1.x

        def residuals_geo_weighted(p):
            xc_, yc_, r_ = p
            ri_ = np.sqrt((x - xc_) ** 2 + (y - yc_) ** 2)
            w = 1.0 / np.maximum(ri_, 1e-12)  # EPS_MIN
            return w * (ri_ - r_)

        res2 = least_squares(residuals_geo_weighted, np.array([xc, yc, r]), method="trf")
        xc, yc, r = res2.x

    ri = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)
    
    # 绘图功能
    if plot:
        _plot_circle_fit(x, y, xc, yc, r, ri)
    
    return float(xc), float(yc), float(abs(r)), (ri - abs(r))

def _plot_circle_fit(x: np.ndarray, y: np.ndarray, xc: float, yc: float, r: float, residuals: np.ndarray):
    """
    绘制圆拟合结果
    
    参数:
        x, y: 原始数据点坐标
        xc, yc, r: 拟合圆的圆心和半径
        residuals: 残差
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 子图1：数据点和拟合圆
    ax1.scatter(x, y, c='blue', s=30, alpha=0.7, label='data point')
    ax1.scatter(xc, yc, c='red', s=50, marker='x', linewidth=2, label='circle center')
    
    # 绘制拟合圆
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = xc + r * np.cos(theta)
    circle_y = yc + r * np.sin(theta)
    ax1.plot(circle_x, circle_y, 'r-', linewidth=2, label=f'fitting circle (r={r:.3f})')
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('circle fitting result')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # 子图2：残差分析
    ax2.scatter(range(len(residuals)), residuals, c='green', s=40, alpha=0.7)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax2.set_xlabel('data point index')
    ax2.set_ylabel('residual')
    ax2.set_title('fitting residual')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 打印拟合信息
    print(f"fitting circle center: ({xc:.6f}, {yc:.6f})")
    print(f"fitting radius: {r:.6f}")
    print(f"RMS: {np.sqrt(np.mean(residuals**2)):.6f}")
    print(f"maxium residual: {np.max(np.abs(residuals)):.6f}")