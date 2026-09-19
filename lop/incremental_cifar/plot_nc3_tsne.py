#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NC3 数据 t-SNE 可视化（incremental_cifar 版）

对每个特征向量做 L2 归一化后，把类中心(○)、分类器权重(□)、样本点(·) 一起做 t-SNE 降维，
绘制类中心与分类器权重的连线，观察二者是否对齐（NC3 自对偶性）。

所有选中的任务一次性做 t-SNE 降维，保证各任务的坐标系一致，便于横向比较。

用法:
  python plot_nc3_tsne.py                          # 可视化全部任务
  python plot_nc3_tsne.py --task_indices 0 5 10    # 只可视化指定任务
  python plot_nc3_tsne.py --combine both           # 额外生成所有任务的组合大图
"""

import os
import sys
import argparse
import pickle
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 默认数据目录：<脚本所在目录>/result/task_continual_learning
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SAVE_DIR = os.path.join(HERE, "result", "task_continual_learning")


def l2_normalize_rows(data):
    """对矩阵每一行做 L2 归一化（范数下限 1e-12 防止除零）"""
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return data / norms


def load_task(nc3_data_dir, task_idx):
    """读取单个任务的 NC3 pickle，返回 L2 归一化后的中心/权重/样本/标签"""
    path = os.path.join(nc3_data_dir, "nc3_data_task_{0}.pkl".format(task_idx))
    with open(path, "rb") as f:
        d = pickle.load(f)
    return {
        "centers": l2_normalize_rows(d["class_centers"]),       # [K, D]
        "weights": l2_normalize_rows(d["classifier_weights"]),  # [K, D]
        "samples": l2_normalize_rows(d["sample_features"]),     # [N, D]
        "labels": d["sample_labels"],                           # [N]
    }


def plot_task(task_idx, centers_2d, weights_2d, samples_2d, labels,
              x_lim, y_lim, save_dir, n_classes):
    """为单个任务绘制「完整图」与「关键点图」两张图"""
    cmap = plt.cm.tab10 if n_classes <= 10 else plt.cm.tab20
    colors = cmap(np.linspace(0, 1, n_classes))
    vis_dir = os.path.join(save_dir, "nc3_visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    global_center = centers_2d.mean(axis=0)
    global_weight = weights_2d.mean(axis=0)

    # ================= 图1：完整图（含样本点） =================
    plt.figure(figsize=(14, 12))

    for c in range(n_classes):
        mask = (labels == c)
        pts = samples_2d[mask]
        if len(pts) > 0:
            plt.scatter(pts[:, 0], pts[:, 1], color=colors[c], s=20, marker=".",
                        alpha=0.3, rasterized=True,
                        label="Class {0} Samples".format(c) if c == 0 else "")
    for c in range(n_classes):
        plt.scatter(centers_2d[c, 0], centers_2d[c, 1], color=colors[c], s=200,
                    marker="o", edgecolors="black", linewidth=2.5, zorder=10,
                    label="Class {0} Center".format(c) if c == 0 else "")
        plt.scatter(weights_2d[c, 0], weights_2d[c, 1], color=colors[c], s=200,
                    marker="s", edgecolors="black", linewidth=2.5, zorder=10,
                    label="Class {0} Weight".format(c) if c == 0 else "")
        plt.plot([centers_2d[c, 0], weights_2d[c, 0]],
                 [centers_2d[c, 1], weights_2d[c, 1]],
                 color=colors[c], alpha=0.7, linewidth=2, zorder=5)
    for c in range(n_classes):
        mask = (labels == c)
        if np.sum(mask) > 0:
            center = samples_2d[mask].mean(axis=0)
            plt.scatter(center[0], center[1], color=colors[c], s=80, marker="*",
                        edgecolors="black", linewidth=1, zorder=8)

    plt.scatter(global_center[0], global_center[1], color="red", marker="*", s=400,
                edgecolors="black", linewidth=3, zorder=15, label="Global Center Mean")
    plt.scatter(global_weight[0], global_weight[1], color="blue", marker="*", s=400,
                edgecolors="black", linewidth=3, zorder=15, label="Global Weight Mean")
    plt.plot([global_center[0], global_weight[0]],
             [global_center[1], global_weight[1]],
             color="black", alpha=0.8, linewidth=3, linestyle="--",
             label="Global Center-Weight")

    plt.title("Task {0} NC3 Visualization (L2-normalized, with samples)\n"
              "Class Centers (circle), Weights (square), Samples (dot)".format(task_idx),
              fontsize=16, fontweight="bold")
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    handles, labels_ = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels_, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=10, loc="best", ncol=2, framealpha=0.9)
    plt.grid(alpha=0.2, linestyle="--")
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "task_{0}_nc3_full_tsne.png".format(task_idx)),
                dpi=300, bbox_inches="tight")
    plt.close()

    # ================= 图2：关键点图（仅中心+权重） =================
    plt.figure(figsize=(14, 12))

    for c in range(n_classes):
        plt.scatter(centers_2d[c, 0], centers_2d[c, 1], color=colors[c], s=300,
                    marker="o", edgecolors="black", linewidth=2.5, zorder=10,
                    label="Class {0} Center".format(c) if c == 0 else "")
        plt.scatter(weights_2d[c, 0], weights_2d[c, 1], color=colors[c], s=300,
                    marker="s", edgecolors="black", linewidth=2.5, zorder=10,
                    label="Class {0} Weight".format(c) if c == 0 else "")
        plt.plot([centers_2d[c, 0], weights_2d[c, 0]],
                 [centers_2d[c, 1], weights_2d[c, 1]],
                 color=colors[c], alpha=0.8, linewidth=2.5, zorder=5)

    plt.scatter(global_center[0], global_center[1], color="red", marker="*", s=500,
                edgecolors="black", linewidth=3, zorder=15, label="Global Center Mean")
    plt.scatter(global_weight[0], global_weight[1], color="blue", marker="*", s=500,
                edgecolors="black", linewidth=3, zorder=15, label="Global Weight Mean")
    plt.plot([global_center[0], global_weight[0]],
             [global_center[1], global_weight[1]],
             color="black", alpha=0.8, linewidth=3, linestyle="--",
             label="Global Center-Weight")

    plt.title("Task {0} NC3 Visualization (L2-normalized, keypoints only)\n"
              "Class Centers (circle), Weights (square)".format(task_idx),
              fontsize=16, fontweight="bold")
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    handles, labels_ = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels_, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=10, loc="best", ncol=2, framealpha=0.9)
    plt.grid(alpha=0.2, linestyle="--")
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "task_{0}_nc3_keyonly_tsne.png".format(task_idx)),
                dpi=300, bbox_inches="tight")
    plt.close()

    print("  [OK] Task {0}: full + keyonly 已保存".format(task_idx))


def plot_combined(tasks, mode, x_lim, y_lim, save_dir, n_classes):
    """把所有任务放进一张组合大图（full 或 keyonly）"""
    n_tasks = len(tasks)
    n_cols = int(np.ceil(np.sqrt(n_tasks)))
    n_rows = int(np.ceil(n_tasks / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    cmap = plt.cm.tab10 if n_classes <= 10 else plt.cm.tab20
    colors = cmap(np.linspace(0, 1, n_classes))

    for idx, task in enumerate(tasks):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]
        centers_2d, weights_2d = task["centers_2d"], task["weights_2d"]

        if mode == "full":
            labels = task["labels"]
            samples_2d = task["samples_2d"]
            for c in range(n_classes):
                mask = (labels == c)
                pts = samples_2d[mask]
                if len(pts) > 0:
                    ax.scatter(pts[:, 0], pts[:, 1], color=colors[c], s=5,
                               marker=".", alpha=0.2, rasterized=True)

        for c in range(n_classes):
            ax.scatter(centers_2d[c, 0], centers_2d[c, 1], color=colors[c], s=50,
                       marker="o", edgecolors="black", linewidth=1, zorder=10)
            ax.scatter(weights_2d[c, 0], weights_2d[c, 1], color=colors[c], s=50,
                       marker="s", edgecolors="black", linewidth=1, zorder=10)
            ax.plot([centers_2d[c, 0], weights_2d[c, 0]],
                    [centers_2d[c, 1], weights_2d[c, 1]],
                    color=colors[c], alpha=0.7, linewidth=1, zorder=5)

        if mode == "full":
            for c in range(n_classes):
                mask = (task["labels"] == c)
                if np.sum(mask) > 0:
                    center = task["samples_2d"][mask].mean(axis=0)
                    ax.scatter(center[0], center[1], color=colors[c], s=20,
                               marker="*", edgecolors="black", linewidth=0.5, zorder=8)

        gc = centers_2d.mean(axis=0)
        gw = weights_2d.mean(axis=0)
        ax.scatter(gc[0], gc[1], color="red", marker="*", s=100,
                   edgecolors="black", linewidth=1, zorder=15)
        ax.scatter(gw[0], gw[1], color="blue", marker="*", s=100,
                   edgecolors="black", linewidth=1, zorder=15)
        ax.plot([gc[0], gw[0]], [gc[1], gw[1]], color="black",
                alpha=0.8, linewidth=1.5, linestyle="--")

        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_aspect("equal")
        ax.set_title("Task {0}".format(task["task_idx"]), fontsize=10)

        if idx % 5 == 0:
            ax.set_xlabel("t-SNE 1", fontsize=8)
            ax.set_ylabel("t-SNE 2", fontsize=8)
        else:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xlabel("")
            ax.set_ylabel("")
        ax.grid(alpha=0.2, linestyle="--")

    for idx in range(n_tasks, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].axis("off")

    plt.suptitle("NC3 t-SNE Visualization ({0} mode) - All Tasks".format(mode),
                 fontsize=16, fontweight="bold")
    plt.tight_layout()
    vis_dir = os.path.join(save_dir, "nc3_visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    plt.savefig(os.path.join(vis_dir, "combined_{0}_tsne.png".format(mode)),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("  [OK] 组合图（{0}）已保存".format(mode))


def main():
    parser = argparse.ArgumentParser(description="NC3 数据 t-SNE 可视化（incremental_cifar）")
    parser.add_argument("--save_dir", type=str, default=DEFAULT_SAVE_DIR,
                        help="数据保存目录（包含 nc3_saved_data 文件夹）")
    parser.add_argument("--task_indices", type=int, nargs="+", default=None,
                        help="要可视化的任务索引，例如 --task_indices 0 5 10；不指定则全部")
    parser.add_argument("--perplexity", type=int, default=None,
                        help="t-SNE 的 perplexity（默认按总点数自动调整）")
    parser.add_argument("--random_seed", type=int, default=42, help="随机种子（默认 42）")
    parser.add_argument("--combine", type=str, choices=["full", "keyonly", "both"],
                        default=None, help="组合大图模式")
    args = parser.parse_args()

    nc3_dir = os.path.join(args.save_dir, "nc3_saved_data")
    if not os.path.isdir(nc3_dir):
        print("[ERROR] NC3 数据目录不存在: {0}".format(nc3_dir))
        sys.exit(1)

    files = glob.glob(os.path.join(nc3_dir, "nc3_data_task_*.pkl"))
    if not files:
        print("[ERROR] 未找到 NC3 数据文件")
        sys.exit(1)

    available = sorted(int(f.split("task_")[-1].split(".pkl")[0]) for f in files)
    task_indices = available if args.task_indices is None else [t for t in args.task_indices if t in available]
    if not task_indices:
        print("[ERROR] 没有有效的任务索引")
        sys.exit(1)

    print("开始全局 t-SNE 降维，共 {0} 个任务: {1}".format(len(task_indices), task_indices))

    # 1. 载入并 L2 归一化所有任务数据
    tasks = [load_task(nc3_dir, t) for t in task_indices]
    n_classes = tasks[0]["centers"].shape[0]
    feat_dim = tasks[0]["centers"].shape[1]

    # 2. 拼接成一个大矩阵：每任务 [centers; weights; samples]
    total_points = sum(t["centers"].shape[0] + t["weights"].shape[0] + t["samples"].shape[0] for t in tasks)
    all_points = np.zeros((total_points, feat_dim), dtype=np.float32)
    boundaries = []
    idx = 0
    for t in tasks:
        start = idx
        for part in ("centers", "weights", "samples"):
            arr = t[part]
            all_points[idx:idx + arr.shape[0]] = arr
            idx += arr.shape[0]
        boundaries.append((start, idx))

    # 3. t-SNE 降维
    perplexity = args.perplexity or min(50, max(5, total_points // 30))
    print("  总点数 {0}, perplexity = {1}".format(total_points, perplexity))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=args.random_seed,
                init="random", learning_rate="auto", max_iter=1000)
    all_points_2d = tsne.fit_transform(all_points)

    x_min, x_max = all_points_2d[:, 0].min(), all_points_2d[:, 0].max()
    y_min, y_max = all_points_2d[:, 1].min(), all_points_2d[:, 1].max()
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1
    x_lim = (x_min - x_margin, x_max + x_margin)
    y_lim = (y_min - y_margin, y_max + y_margin)

    # 4. 逐任务切分并绘图
    combined_full, combined_key = [], []
    for i, task_idx in enumerate(task_indices):
        start, end = boundaries[i]
        pts = all_points_2d[start:end]
        ncls = tasks[i]["centers"].shape[0]
        centers_2d = pts[:ncls]
        weights_2d = pts[ncls:2 * ncls]
        samples_2d = pts[2 * ncls:]

        plot_task(task_idx, centers_2d, weights_2d, samples_2d, tasks[i]["labels"],
                  x_lim, y_lim, args.save_dir, n_classes)

        combined_full.append({"task_idx": task_idx, "centers_2d": centers_2d,
                              "weights_2d": weights_2d, "samples_2d": samples_2d,
                              "labels": tasks[i]["labels"]})
        combined_key.append({"task_idx": task_idx, "centers_2d": centers_2d,
                             "weights_2d": weights_2d})

    if args.combine in ("full", "both"):
        plot_combined(combined_full, "full", x_lim, y_lim, args.save_dir, n_classes)
    if args.combine in ("keyonly", "both"):
        plot_combined(combined_key, "keyonly", x_lim, y_lim, args.save_dir, n_classes)

    print("全部任务可视化完成，图片保存在 {0}/nc3_visualizations/".format(args.save_dir))


if __name__ == "__main__":
    main()
