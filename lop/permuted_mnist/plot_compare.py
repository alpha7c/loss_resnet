#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多算法可塑性指标对比可视化模块
读取四个算法（BP, EWC, CBP, L2）的 plasticity.csv 文件，绘制：
1. 损失下降量 (L_pre - L_post) 对比
2. 可塑性指标 P_n 对比
3. 参数变化量 (||Δθ||) 对比
图像保存为 plasticity_metrics_comparison.png
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==================== 配置区域（请根据实际情况修改） ====================
# 四个算法的配置文件列表：每个元素为 (算法名称, CSV文件路径, 最大任务数)
ALGORITHMS = [
    ("BP",   "/home/jxqi/PL_NC/lop/permuted_mnist/bp_cifar/plasticity_metrics.csv", 10),
    ("EWC",  "/home/jxqi/PL_NC/lop/permuted_mnist/ewc_duo/plasticity.csv", 10),
    ("CBP",  "/home/jxqi/PL_NC/lop/permuted_mnist/cbp_cifar/plasticity_metrics.csv", 10),
    ("L2",   "/home/jxqi/PL_NC/lop/permuted_mnist/l2_cifar/plasticity_metrics.csv", 10),  # 新增L2算法
]

# 输出配置
OUTPUT_DIR = "/home/jxqi/PL_NC/lop/permuted_mnist/comparison"   # 图像保存目录
OUTPUT_FILENAME = "plasticity_metrics_comparison.png"           # 输出图像文件名
# =======================================================================

# 颜色与样式映射（按算法顺序）
COLORS = {
    "BP":  "green",
    "EWC": "blue",
    "CBP": "red",
    "L2":  "purple",       # 为L2分配紫色
}
MARKERS = {
    "BP":  "o",
    "EWC": "s",
    "CBP": "^",
    "L2":  "D",            # 菱形标记，与其它算法区分
}
LINESTYLES = {
    "BP":  "-",
    "EWC": "--",
    "CBP": ":",
    "L2":  "-.",           # 点划线，增强区分度
}

def load_algorithm_data(csv_path, max_tasks):
    """加载单个算法的数据，返回 (task_idx, loss_drop, P_n, delta_norm) 列表，长度等于 max_tasks，缺失部分用 NaN 填充"""
    if not os.path.exists(csv_path):
        print(f"警告：文件不存在 {csv_path}")
        return None
    df = pd.read_csv(csv_path)
    required_cols = ['task_idx', 'loss_pre', 'loss_post', 'delta_norm_total', 'P_n']
    for col in required_cols:
        if col not in df.columns:
            print(f"错误：{csv_path} 中缺少列 '{col}'")
            return None
    
    # 提取数据，按 task_idx 排序
    df = df.sort_values('task_idx')
    actual_tasks = df['task_idx'].values
    loss_drop = (df['loss_pre'] - df['loss_post']).values
    P_n = df['P_n'].values
    delta_norm = df['delta_norm_total'].values
    
    # 创建长度为 max_tasks 的数组，缺失部分填 np.nan
    full_task_idx = np.arange(max_tasks)
    full_loss_drop = np.full(max_tasks, np.nan)
    full_P_n = np.full(max_tasks, np.nan)
    full_delta_norm = np.full(max_tasks, np.nan)
    
    # 将已有数据填充到对应索引位置
    for idx, t in enumerate(actual_tasks):
        if t < max_tasks:
            full_loss_drop[t] = loss_drop[idx]
            full_P_n[t] = P_n[idx]
            full_delta_norm[t] = delta_norm[idx]
    
    return full_task_idx, full_loss_drop, full_P_n, full_delta_norm

def main():
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 确定全局最大任务数（用于横坐标范围）
    max_tasks_global = max(cfg[2] for cfg in ALGORITHMS)
    x_ticks = np.arange(max_tasks_global)
    
    # 加载所有算法数据
    data_dict = {}
    for name, path, max_tasks in ALGORITHMS:
        result = load_algorithm_data(path, max_tasks_global)  # 统一使用全局最大任务数
        if result is not None:
            data_dict[name] = result  # (task_idx, loss_drop, P_n, delta_norm)
        else:
            print(f"跳过算法 {name}，数据加载失败")
    
    if not data_dict:
        print("没有成功加载任何算法数据，退出。")
        return
    
    # 设置绘图风格
    #plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)
    
    # 子图1：损失下降量 (分子)
    ax1 = axes[0]
    for name, (_, loss_drop, _, _) in data_dict.items():
        ax1.plot(x_ticks, loss_drop, 
                 marker=MARKERS.get(name, 'o'), 
                 linestyle=LINESTYLES.get(name, '-'), 
                 color=COLORS.get(name, 'gray'),
                 linewidth=2, markersize=6, label=name)
    ax1.set_ylabel(r'Loss Drop ($\mathcal{L}_{pre} - \mathcal{L}_{post}$)', fontsize=12)
    ax1.set_title('Loss Reduction After One Epoch (Comparison)', fontsize=14)
    ax1.set_ylim(0, 6)
    ax1.set_yticks(np.arange(0, 6.1, 1))
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    
    # 子图2：可塑性指标 P_n
    ax2 = axes[1]
    for name, (_, _, P_n, _) in data_dict.items():
        ax2.plot(x_ticks, P_n,
                 marker=MARKERS.get(name, 'o'),
                 linestyle=LINESTYLES.get(name, '-'),
                 color=COLORS.get(name, 'gray'),
                 linewidth=2, markersize=6, label=name)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax2.set_ylabel(r'$P_n$', fontsize=12)
    ax2.set_title('Plasticity Metric $P_n$ vs Task Index', fontsize=14)
    ax2.set_ylim(0, 0.5)
    ax2.set_yticks(np.arange(0, 0.51, 0.05))
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')
    
    # 子图3：参数变化量 (分母)
    ax3 = axes[2]
    for name, (_, _, _, delta_norm) in data_dict.items():
        ax3.plot(x_ticks, delta_norm,
                 marker=MARKERS.get(name, 's'),
                 linestyle=LINESTYLES.get(name, '-'),
                 color=COLORS.get(name, 'gray'),
                 linewidth=2, markersize=6, label=name)
    ax3.set_xlabel('Task Index', fontsize=12)
    ax3.set_ylabel(r'$||\Delta \theta||$', fontsize=12)
    ax3.set_title('Parameter Change Norm After One Epoch', fontsize=14)
    ax3.set_ylim(0, 35)
    ax3.set_yticks(np.arange(0, 36, 5))
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='best')
    
    # 设置x轴刻度
    ax3.set_xticks(x_ticks)
    ax3.set_xticklabels([str(i) for i in x_ticks])
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    save_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"多算法可塑性指标对比图像已保存至: {save_path}")
    
    plt.close()

if __name__ == "__main__":
    main()