#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可塑性指标可视化模块
读取 plasticity.csv 文件，绘制：
1. 损失下降量 (L_pre - L_post) 随任务索引的变化
2. 可塑性指标 P_n 随任务索引的变化
3. 参数变化量 (||Δθ||) 随任务索引的变化
图像保存为 plasticity_metrics.png
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==================== 配置区域（请根据实际情况修改） ====================
CSV_PATH = "/home/jxqi/PL_NC/lop/permuted_mnist/6/plasticity.csv"          # plasticity.csv 文件路径
OUTPUT_DIR = "/home/jxqi/PL_NC/lop/permuted_mnist/6"               # 图像保存目录
OUTPUT_FILENAME = "plasticity_metrics.png"  # 输出图像文件名
# =======================================================================

def main():
    # 检查文件是否存在
    if not os.path.exists(CSV_PATH):
        print(f"错误：找不到文件 {CSV_PATH}")
        return
    
    # 读取数据
    df = pd.read_csv(CSV_PATH)
    
    # 确保必需的列存在
    required_cols = ['task_idx', 'loss_pre', 'loss_post', 'delta_norm_total', 'P_n']
    for col in required_cols:
        if col not in df.columns:
            print(f"错误：CSV 文件中缺少列 '{col}'")
            return
    
    # 提取数据
    task_idx = df['task_idx'].values
    P_n = df['P_n'].values
    loss_drop = df['loss_pre'] - df['loss_post']   # 损失下降量
    delta_norm = df['delta_norm_total'].values
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 设置绘图风格
    plt.style.use('seaborn-v0_8-darkgrid')
    # 调整子图顺序：分子 -> 指标 -> 分母
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # 子图1：损失下降量 (分子) - 折线图
    ax1 = axes[0]
    ax1.plot(task_idx, loss_drop, marker='o', linestyle='-', color='g', linewidth=2, markersize=6)
    ax1.set_ylabel(r'Loss Drop ($\mathcal{L}_{pre} - \mathcal{L}_{post}$)', fontsize=12)
    ax1.set_title('Loss Reduction After One Epoch', fontsize=14)
    ax1.set_ylim(0, 6)
    ax1.set_yticks(np.arange(0, 6.1, 1))
    ax1.grid(True, alpha=0.3)
    
    # 子图2：可塑性指标 P_n
    ax2 = axes[1]
    ax2.plot(task_idx, P_n, marker='o', linestyle='-', color='b', linewidth=2, markersize=6)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax2.set_ylabel(r'$P_n$', fontsize=12)
    ax2.set_title('Plasticity Metric $P_n$ vs Task Index', fontsize=14)
    ax2.set_ylim(0, 0.25)
    ax2.set_yticks(np.arange(0, 0.261, 0.05))
    ax2.grid(True, alpha=0.3)
    
    # 子图3：参数变化量 (分母)
    ax3 = axes[2]
    ax3.plot(task_idx, delta_norm, marker='s', linestyle='-', color='r', linewidth=2, markersize=6)
    ax3.set_xlabel('Task Index', fontsize=12)
    ax3.set_ylabel(r'$||\Delta \theta||$', fontsize=12)
    ax3.set_title('Parameter Change Norm After One Epoch', fontsize=14)
    ax3.set_ylim(0, 70)
    ax3.set_yticks(np.arange(0, 71, 5))
    ax3.grid(True, alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    save_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"可塑性指标可视化图像已保存至: {save_path}")
    
    plt.close()

if __name__ == "__main__":
    main()