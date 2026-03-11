import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def phase_model(f, theta0, Ql, fr):
    """相位模型：θ(f) = θ₀ + 2·arctan[2·Qₗ·(1 - f/fᵣ)]"""
    return theta0 + 2 * np.arctan(2 * Ql * (1 - f / fr))

def phase_fit(f_data, phase, theta0_guess, Ql_guess, fr_guess, plot=False):
    """
    简化的相位拟合
    
    TEXT
    参数:
    ----------
    f_data : 频率数组
    z_data : 复数数据
    theta0_guess : θ₀的初始猜测
    Ql_guess : Qₗ的初始猜测
    fr_guess : fᵣ的初始猜测
    
    返回:
    ----------
    theta0, Ql, fr : 拟合参数
    """
    
    # 提取相位并展开
    phase_unwrapped = np.unwrap(phase)
    
    # 使用curve_fit直接拟合
    p0 = [theta0_guess, Ql_guess, fr_guess]
    
    # 设置合理的边界
    bounds = (
        [-np.pi*10, 100, f_data[0]],      # 下界
        [np.pi*10, 1e6, f_data[-1]]       # 上界
    )
    
    try:
        popt, pcov = curve_fit(
            phase_model, 
            f_data, 
            phase_unwrapped, 
            p0=p0,
            bounds=bounds,
            maxfev=10000
        )
        theta0, Ql, fr = popt
        
        # Calculate parameter uncertainties
        perr = np.sqrt(np.diag(pcov))
    except Exception as e:
        print(f"拟合失败: {e}")
        # 如果拟合失败，返回初始猜测
        theta0, Ql, fr = theta0_guess, Ql_guess, fr_guess
        perr = [0, 0, 0]
    if plot:
        # Create figure and plot
        plt.figure(figsize=(10, 6))
        phase_fit_curve = phase_model(f_data, theta0, Ql, fr)
        
        plt.plot(f_data/1e9, phase_unwrapped, 'b-', label='Measured Phase', alpha=0.7, linewidth=1.5)
        plt.plot(f_data/1e9, phase_fit_curve, 'r--', linewidth=2, label='Fitted Curve')
        
        # Add fitted parameters as text box
        param_text = f'Fitted Parameters:\n'
        param_text += f'θ = {theta0:.4f} rad\n'
        param_text += f'Ql = {Ql:.2e}\n'
        param_text += f'fr = {fr/1e9:.6f} GHz'
        
        # Add parameter uncertainties if available
        if np.any(perr > 0):
            param_text += f'\n\nUncertainties:\n'
            param_text += f'Δθ = {perr[0]:.4f} rad\n'
            param_text += f'ΔQl = {perr[1]:.2e}\n'
            param_text += f'Δfr = {perr[2]/1e9:.6f} GHz'
        
        # Create a text box with a semi-transparent background
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        plt.text(0.02, 0.98, param_text, transform=plt.gca().transAxes, 
                 fontsize=10, verticalalignment='top', bbox=props)
        
        # Alternatively, place text on the plot at the resonance frequency position
        # Find the index closest to resonance frequency
        idx = np.argmin(np.abs(f_data - fr))
        plt.plot(fr/1e9, phase_unwrapped[idx], 'ro', markersize=8, 
                 label=f'Resonance: {fr/1e9:.6f} GHz')
        
        # Add annotations for key points
        plt.annotate(f'fᵣ = {fr/1e9:.6f} GHz', 
                     xy=(fr/1e9, phase_unwrapped[idx]),
                     xytext=(fr/1e9 + 0.001, phase_unwrapped[idx] + 0.5),
                     arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                     fontsize=9, color='red')
        
        # Formatting
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('Phase (rad)')
        plt.title('Phase Response Fit')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()
        
        # Print parameters to console as well
        print("Fitted Parameters:")
        print(f"θ₀ = {theta0:.6f} rad")
        print(f"Qₗ = {Ql:.2e}")
        print(f"fᵣ = {fr/1e9:.9f} GHz")
        if np.any(perr > 0):
            print(f"\nParameter Uncertainties:")
            print(f"Δθ₀ = {perr[0]:.6f} rad")
            print(f"ΔQₗ = {perr[1]:.2e}")
            print(f"Δfᵣ = {perr[2]/1e9:.9f} GHz")
    
    return theta0, Ql, fr
