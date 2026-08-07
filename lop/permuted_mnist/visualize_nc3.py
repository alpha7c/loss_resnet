#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NC3数据t-SNE可视化独立模块（每个样本向量进行L2归一化）
生成两张图：
- 完整图（含样本点）
- 关键点图（仅类中心+分类器权重，坐标范围与完整图一致）
用法: python visualize_nc3.py --save_dir <保存目录> [--task_indices 0 5 10] [--perplexity 30] [--combine {full,keyonly,both}]

修改：所有任务的数据一次性降维，然后逐个任务绘图（坐标系一致）
新增：组合图模式，将所有任务的图放在一个大图中，子图横坐标每隔5个任务标注一次任务编号
"""

import os
import sys
import argparse
import pickle
import glob
import gc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def l2_normalize_rows(data):
    """对输入矩阵的每一行进行L2归一化"""
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return data / norms


def plot_task_from_global_tsne(task_idx, centers_2d, weights_2d, samples_2d,
                               sample_labels, x_lim, y_lim, save_dir):
    """
    使用已经降维好的坐标绘制单个任务的两张图（独立保存）
    """
    num_classes = centers_2d.shape[0]
    cmap = plt.cm.tab10 if num_classes <= 10 else plt.cm.tab20
    colors = cmap(np.linspace(0, 1, num_classes))
    vis_dir = os.path.join(save_dir, "nc3_visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    # ==================== 图1：完整图 ====================
    plt.figure(figsize=(14, 12))

    # 样本点
    for c in range(num_classes):
        class_mask = (sample_labels == c)
        class_samples = samples_2d[class_mask]
        if len(class_samples) > 0:
            plt.scatter(class_samples[:, 0], class_samples[:, 1],
                        color=colors[c], s=20, marker='.', alpha=0.3,
                        label=f"Class {c} Samples" if c == 0 else "",
                        rasterized=True)

    # 类中心（圆）
    for c in range(num_classes):
        plt.scatter(centers_2d[c, 0], centers_2d[c, 1],
                    color=colors[c], s=200, marker='o',
                    edgecolors='black', linewidth=2.5, zorder=10,
                    label=f"Class {c} Center" if c == 0 else "")

    # 分类器权重（方块）
    for c in range(num_classes):
        plt.scatter(weights_2d[c, 0], weights_2d[c, 1],
                    color=colors[c], s=200, marker='s',
                    edgecolors='black', linewidth=2.5, zorder=10,
                    label=f"Class {c} Weight" if c == 0 else "")

    # 连线
    for c in range(num_classes):
        plt.plot([centers_2d[c, 0], weights_2d[c, 0]],
                 [centers_2d[c, 1], weights_2d[c, 1]],
                 color=colors[c], alpha=0.7, linewidth=2, linestyle='-', zorder=5)

    # 样本中心（星号）
    for c in range(num_classes):
        class_mask = (sample_labels == c)
        if np.sum(class_mask) > 0:
            center = samples_2d[class_mask].mean(axis=0)
            plt.scatter(center[0], center[1],
                        color=colors[c], s=80, marker='*',
                        edgecolors='black', linewidth=1, zorder=8)
    
    # 全局平均值（基于本任务）
    global_center = centers_2d.mean(axis=0)
    global_weight = weights_2d.mean(axis=0)
    plt.scatter(global_center[0], global_center[1], color='red', marker='*', s=400,
                edgecolors='black', linewidth=3, zorder=15, label="Global Center Mean")
    plt.scatter(global_weight[0], global_weight[1], color='blue', marker='*', s=400,
                edgecolors='black', linewidth=3, zorder=15, label="Global Weight Mean")
    plt.plot([global_center[0], global_weight[0]],
             [global_center[1], global_weight[1]],
             color='black', alpha=0.8, linewidth=3, linestyle='--',
             label="Global Center ↔ Weight")
    
    plt.title(f"Task {task_idx} NC3 Visualization (L2-normalized, with samples)\nClass Centers (○), Weights (□), Samples (·)",
              fontsize=16, fontweight='bold')
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=10, loc='best', ncol=2, framealpha=0.9)
    plt.grid(alpha=0.2, linestyle='--')
    plt.tight_layout()
    
    save_path_full = os.path.join(vis_dir, f"task_{task_idx}_nc3_full_tsne.png")
    plt.savefig(save_path_full, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ 完整图已保存: {save_path_full}")
    
    # ==================== 图2：关键点图 ====================
    plt.figure(figsize=(14, 12))
    
    for c in range(num_classes):
        plt.scatter(centers_2d[c, 0], centers_2d[c, 1],
                    color=colors[c], s=300, marker='o',
                    edgecolors='black', linewidth=2.5, zorder=10,
                    label=f"Class {c} Center" if c == 0 else "")

    for c in range(num_classes):
        plt.scatter(weights_2d[c, 0], weights_2d[c, 1],
                    color=colors[c], s=300, marker='s',
                    edgecolors='black', linewidth=2.5, zorder=10,
                    label=f"Class {c} Weight" if c == 0 else "")

    for c in range(num_classes):
        plt.plot([centers_2d[c, 0], weights_2d[c, 0]],
                 [centers_2d[c, 1], weights_2d[c, 1]],
                 color=colors[c], alpha=0.8, linewidth=2.5, linestyle='-', zorder=5)

    plt.scatter(global_center[0], global_center[1], color='red', marker='*', s=500,
                edgecolors='black', linewidth=3, zorder=15, label="Global Center Mean")
    plt.scatter(global_weight[0], global_weight[1], color='blue', marker='*', s=500,
                edgecolors='black', linewidth=3, zorder=15, label="Global Weight Mean")
    plt.plot([global_center[0], global_weight[0]],
             [global_center[1], global_weight[1]],
             color='black', alpha=0.8, linewidth=3, linestyle='--',
             label="Global Center ↔ Weight")

    plt.title(f"Task {task_idx} NC3 Visualization (L2-normalized, keypoints only)\nClass Centers (○), Weights (□)",
              fontsize=16, fontweight='bold')
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=10, loc='best', ncol=2, framealpha=0.9)
    plt.grid(alpha=0.2, linestyle='--')
    plt.tight_layout()
    
    save_path_key = os.path.join(vis_dir, f"task_{task_idx}_nc3_keyonly_tsne.png")
    plt.savefig(save_path_key, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ 关键点图已保存: {save_path_key}")


def combine_tasks_figure(task_data_list, plot_type, x_lim, y_lim, save_dir):
    """
    将所有任务的图组合到一个大图中
    task_data_list: list of dict, 每个元素包含:
        task_idx, centers_2d, weights_2d, samples_2d, sample_labels
    plot_type: 'full' 或 'keyonly'
    x_lim, y_lim: 统一的坐标范围
    save_dir: 保存目录
    """
    n_tasks = len(task_data_list)
    # 动态计算子图布局：尽量接近正方形
    n_cols = int(np.ceil(np.sqrt(n_tasks)))
    n_rows = int(np.ceil(n_tasks / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    # 如果 axes 是一维的，统一处理为二维数组
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    first_task = task_data_list[0]
    num_classes = first_task['centers_2d'].shape[0]
    cmap = plt.cm.tab10 if num_classes <= 10 else plt.cm.tab20
    colors = cmap(np.linspace(0, 1, num_classes))
    
    for idx, task in enumerate(task_data_list):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        task_idx = task['task_idx']
        centers_2d = task['centers_2d']
        weights_2d = task['weights_2d']
        sample_labels = task['sample_labels']
        
        if plot_type == 'full':
            samples_2d = task['samples_2d']
            # 样本点
            for c in range(num_classes):
                class_mask = (sample_labels == c)
                class_samples = samples_2d[class_mask]
                if len(class_samples) > 0:
                    ax.scatter(class_samples[:, 0], class_samples[:, 1],
                               color=colors[c], s=5, marker='.', alpha=0.2, rasterized=True)

        # 类中心（圆）
        for c in range(num_classes):
            ax.scatter(centers_2d[c, 0], centers_2d[c, 1],
                       color=colors[c], s=50, marker='o',
                       edgecolors='black', linewidth=1, zorder=10)

        # 分类器权重（方块）
        for c in range(num_classes):
            ax.scatter(weights_2d[c, 0], weights_2d[c, 1],
                       color=colors[c], s=50, marker='s',
                       edgecolors='black', linewidth=1, zorder=10)

        # 连线
        for c in range(num_classes):
            ax.plot([centers_2d[c, 0], weights_2d[c, 0]],
                    [centers_2d[c, 1], weights_2d[c, 1]],
                    color=colors[c], alpha=0.7, linewidth=1, linestyle='-', zorder=5)

        if plot_type == 'full':
            # 样本中心（星号）
            for c in range(num_classes):
                class_mask = (sample_labels == c)
                if np.sum(class_mask) > 0:
                    center = samples_2d[class_mask].mean(axis=0)
                    ax.scatter(center[0], center[1],
                               color=colors[c], s=20, marker='*',
                               edgecolors='black', linewidth=0.5, zorder=8)
        
        # 全局平均值（基于本任务）
        global_center = centers_2d.mean(axis=0)
        global_weight = weights_2d.mean(axis=0)
        ax.scatter(global_center[0], global_center[1], color='red', marker='*', s=100,
                   edgecolors='black', linewidth=1, zorder=15)
        ax.scatter(global_weight[0], global_weight[1], color='blue', marker='*', s=100,
                   edgecolors='black', linewidth=1, zorder=15)
        ax.plot([global_center[0], global_weight[0]],
                [global_center[1], global_weight[1]],
                color='black', alpha=0.8, linewidth=1.5, linestyle='--')
        
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_aspect('equal')
        
        # 设置子图标题（显示任务编号）
        ax.set_title(f"Task {task_idx}", fontsize=10)
        
        # 每隔5个任务（即索引能被5整除）才显示x轴和y轴标签，否则隐藏
        if idx % 5 == 0:
            ax.set_xlabel("t-SNE 1", fontsize=8)
            ax.set_ylabel("t-SNE 2", fontsize=8)
        else:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xlabel("")
            ax.set_ylabel("")
        
        # 网格线
        ax.grid(alpha=0.2, linestyle='--')
    
    # 隐藏多余的子图
    for idx in range(n_tasks, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.suptitle(f"NC3 t-SNE Visualization ({plot_type} mode) - All Tasks", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存组合图
    vis_dir = os.path.join(save_dir, "nc3_visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    save_path = os.path.join(vis_dir, f"combined_{plot_type}_tsne.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ 组合图已保存: {save_path}")


def visualize_all_tasks_nc3(base_save_dir, task_indices=None, perplexity=30, random_state=42, combine_mode=None):
    """
    批量可视化所有任务的NC3数据：
    1. 加载所有指定任务的数据，L2归一化后合并为一个超大矩阵
    2. 一次性执行t-SNE降维
    3. 按任务切分降维结果，为每个任务绘制两张图（独立保存）
    4. 如果 combine_mode 不为 None，则额外生成组合图（full 和/或 keyonly）
    """
    nc3_data_dir = os.path.join(base_save_dir, "nc3_saved_data")
    if not os.path.exists(nc3_data_dir):
        print(f"❌ NC3数据目录不存在: {nc3_data_dir}")
        return False
    
    data_files = glob.glob(os.path.join(nc3_data_dir, "nc3_data_task_*.pkl"))
    if not data_files:
        print(f"❌ 未找到NC3数据文件")
        return False
    
    available_tasks = []
    for f in data_files:
        try:
            task_idx = int(f.split("task_")[-1].split(".pkl")[0])
            available_tasks.append(task_idx)
        except:
            continue
    available_tasks.sort()
    
    if task_indices is None:
        task_indices = available_tasks
    else:
        task_indices = [t for t in task_indices if t in available_tasks]
    
    if not task_indices:
        print(f"❌ 没有找到有效的任务索引")
        return False
    
    print(f"\n=== 准备全局t-SNE降维，共 {len(task_indices)} 个任务 ===")
    
    # 1. 加载所有任务数据，并记录每个任务的元信息
    all_centers = []      # 存储每个任务的类中心 (10, D)
    all_weights = []      # 存储每个任务的分类器权重 (10, D)
    all_samples = []      # 存储每个任务的样本点 (2000, D)
    all_labels = []       # 存储每个任务的样本标签 (2000,)
    task_boundaries = []  # 记录每个任务在合并矩阵中的起始和结束索引（用于拆分）
    
    # 先加载所有数据，同时进行L2归一化
    for task_idx in task_indices:
        data_path = os.path.join(nc3_data_dir, f"nc3_data_task_{task_idx}.pkl")
        try:
            with open(data_path, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"❌ 加载任务 {task_idx} 失败: {e}")
            return False
        
        centers = data["class_centers"]          # [10, D]
        weights = data["classifier_weights"]     # [10, D]
        samples = data["sample_features"]        # [2000, D]
        labels = data["sample_labels"]           # [2000]
        
        # L2归一化
        centers_norm = l2_normalize_rows(centers)
        weights_norm = l2_normalize_rows(weights)
        samples_norm = l2_normalize_rows(samples)
        
        all_centers.append(centers_norm)
        all_weights.append(weights_norm)
        all_samples.append(samples_norm)
        all_labels.append(labels)
    
    # 2. 合并所有点为一个超大矩阵（每个任务的点数由实际数据决定）
    n_classes = all_centers[0].shape[0]
    n_samples_first = all_samples[0].shape[0]
    total_points = sum(c.shape[0] + w.shape[0] + s.shape[0] for c, w, s in zip(all_centers, all_weights, all_samples))
    print(f"  总点数: {total_points} ({n_classes} classes × {n_samples_first // n_classes} samples/class per task)")
    
    # 预分配矩阵
    all_points = np.zeros((total_points, all_centers[0].shape[1]), dtype=np.float32)
    
    idx = 0
    for i in range(len(task_indices)):
        centers = all_centers[i]
        weights = all_weights[i]
        samples = all_samples[i]
        n_centers = centers.shape[0]
        n_weights = weights.shape[0]
        n_samples = samples.shape[0]
        task_start = idx
        all_points[idx:idx+n_centers] = centers
        idx += n_centers
        all_points[idx:idx+n_weights] = weights
        idx += n_weights
        all_points[idx:idx+n_samples] = samples
        idx += n_samples
        task_boundaries.append((task_start, idx))
    
    # 3. 执行t-SNE降维
    if perplexity is None:
        perplexity_val = min(50, max(5, total_points // 30))
    else:
        perplexity_val = perplexity
    print(f"  使用 perplexity = {perplexity_val}")
    
    tsne_params = {
        'n_components': 2,
        'perplexity': perplexity_val,
        'random_state': random_state,
        'init': 'random',
        'learning_rate': 'auto'
    }
    try:
        tsne = TSNE(**tsne_params, n_iter=1000)
    except TypeError:
        try:
            tsne = TSNE(**tsne_params, max_iter=1000)
        except TypeError:
            tsne = TSNE(**tsne_params)
    
    print("  正在进行t-SNE降维（所有任务一起）...")
    all_points_2d = tsne.fit_transform(all_points)
    print("  降维完成")
    
    # 4. 计算全局统一的坐标范围
    x_min, x_max = all_points_2d[:, 0].min(), all_points_2d[:, 0].max()
    y_min, y_max = all_points_2d[:, 1].min(), all_points_2d[:, 1].max()
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1
    x_lim = (x_min - x_margin, x_max + x_margin)
    y_lim = (y_min - y_margin, y_max + y_margin)
    
    # 5. 按任务切分并绘图（独立保存）
    print(f"\n=== 开始为每个任务绘图（共 {len(task_indices)} 个任务） ===")
    # 如果需要组合图，先收集每个任务的降维结果
    combine_data_full = []   # 存储用于组合图的数据（full模式）
    combine_data_key = []    # 存储用于组合图的数据（keyonly模式）
    
    for i, task_idx in enumerate(task_indices):
        print(f"\n处理 Task {task_idx}...")
        start, end = task_boundaries[i]
        task_points = all_points_2d[start:end]
        
        ncls = all_centers[i].shape[0]
        centers_2d = task_points[:ncls]
        weights_2d = task_points[ncls:2*ncls]
        samples_2d = task_points[2*ncls:]
        
        # 独立保存图片（原有功能）
        plot_task_from_global_tsne(
            task_idx=task_idx,
            centers_2d=centers_2d,
            weights_2d=weights_2d,
            samples_2d=samples_2d,
            sample_labels=all_labels[i],
            x_lim=x_lim,
            y_lim=y_lim,
            save_dir=base_save_dir
        )
        
        # 收集组合图所需数据
        combine_data_full.append({
            'task_idx': task_idx,
            'centers_2d': centers_2d,
            'weights_2d': weights_2d,
            'samples_2d': samples_2d,
            'sample_labels': all_labels[i]
        })
        combine_data_key.append({
            'task_idx': task_idx,
            'centers_2d': centers_2d,
            'weights_2d': weights_2d,
            'sample_labels': all_labels[i]
        })
    
    # 6. 生成组合图（如果指定）
    if combine_mode in ['full', 'both']:
        print("\n=== 生成组合图（full模式） ===")
        combine_tasks_figure(combine_data_full, 'full', x_lim, y_lim, base_save_dir)
    if combine_mode in ['keyonly', 'both']:
        print("\n=== 生成组合图（keyonly模式） ===")
        combine_tasks_figure(combine_data_key, 'keyonly', x_lim, y_lim, base_save_dir)
    
    # 释放内存
    del all_points, all_points_2d
    gc.collect()
    
    print(f"\n✅ 所有任务可视化完成！")
    return True


def main():
    parser = argparse.ArgumentParser(description='NC3数据t-SNE可视化工具（全局降维，逐个任务绘图）')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='数据保存目录（包含nc3_saved_data文件夹）')
    parser.add_argument('--task_indices', type=int, nargs='+', default=None,
                        help='要可视化的任务索引列表，例如: --task_indices 0 5 10，不指定则可视化所有')
    parser.add_argument('--perplexity', type=int, default=None,
                        help='t-SNE的perplexity参数（默认根据总点数自动调整）')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='随机种子（默认42）')
    parser.add_argument('--combine', type=str, choices=['full', 'keyonly', 'both'], default=None,
                        help='组合图模式：将多个任务的图放在一个大图中。full=完整图，keyonly=关键点图，both=两者都生成')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.save_dir):
        print(f"❌ 目录不存在: {args.save_dir}")
        sys.exit(1)
    
    success = visualize_all_tasks_nc3(
        base_save_dir=args.save_dir,
        task_indices=args.task_indices,
        perplexity=args.perplexity,
        random_state=args.random_seed,
        combine_mode=args.combine
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()