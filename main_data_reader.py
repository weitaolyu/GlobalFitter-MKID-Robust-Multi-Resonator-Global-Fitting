
"""
数据读取工具模块
提供读取S2P文件和文本格式数据的功能
"""

import numpy as np
import re
from typing import Tuple

# 常量定义
DB_SCALE = 20.0  # dB缩放系数

def read_s2p(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    读取S2P格式文件（使用scikit-rf包）
    
    Args:
        filename: S2P文件路径
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (频率数组, S21复数数组)
        
    Raises:
        ImportError: 如果未安装scikit-rf
        ValueError: 文件读取错误
    """
    try:
        import skrf as rf
    except ImportError:
        raise ImportError(
            "请安装scikit-rf包: pip install scikit-rf\n"
            "或使用: conda install -c conda-forge scikit-rf"
        )
    
    try:
        # 使用scikit-rf读取S2P文件
        network = rf.Network(filename)
        
        # 获取频率和S21参数
        frequency = network.f  # 频率数组，单位Hz
        s21_complex = network.s[:, 1, 0]  # S21参数，复数形式
        
        return frequency, s21_complex
        
    except Exception as e:
        raise ValueError(f"无法读取S2P文件 '{filename}': {e}")


def read_txt(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    读取文本格式数据文件
    
    支持格式：第一列频率(Hz)，第二列S21复数形式
    
    Args:
        filename: 文本文件路径
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (频率数组, S21复数数组)
        
    Raises:
        ValueError: 文件格式错误或无法读取数据
    """
    try:
        # 直接使用numpy加载数据
        data = np.loadtxt(filename, dtype=complex, comments='#')
        
        if data.shape[1] < 2:
            raise ValueError("数据文件列数不足，需要至少2列数据")
        
        # 第一列是频率（Hz），转换为实数
        freqs = np.real(data[:, 0])
        
        # 第二列是S21复数
        s21_list = data[:, 1]
        
        return freqs, s21_list
        
    except Exception as e:
        # 如果numpy直接加载失败，尝试逐行读取
        try:
            freqs, s21_list = [], []
            with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    # 跳过空行和注释行
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    
                    try:
                        # 第一列频率（实数）
                        freq_val = float(parts[0])
                        # 第二列S21复数
                        s21_complex = complex(parts[1])
                        
                        freqs.append(freq_val)
                        s21_list.append(s21_complex)
                    except (ValueError, IndexError):
                        continue
            
            if len(freqs) == 0:
                raise ValueError("未能从文件中读取到有效数据")
                
            return np.array(freqs), np.array(s21_list)
            
        except Exception as inner_e:
            raise ValueError(f"读取数据文件失败: {inner_e}")


def read_data(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    通用数据读取函数，自动检测文件格式
    
    Args:
        filename: 数据文件路径
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (频率数组, S21复数数组)
        
    Raises:
        ValueError: 文件格式不被支持或读取失败
        FileNotFoundError: 文件不存在
    """
    if not filename.lower().endswith('.s2p'):
        # 非S2P文件，使用文本格式读取
        return read_txt(filename)
    else:
        # S2P文件
        return read_s2p(filename)


# 导出列表
__all__ = ['read_s2p', 'read_txt', 'read_data', 'DB_SCALE']
