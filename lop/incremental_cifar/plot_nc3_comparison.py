import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===================== 修复字体警告 + 论文级绘图风格 =====================
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.dpi"] = 300

# ===================== 路径配置 =====================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# ===================== 论文级配色方案（与 permuted_mnist 保持一致） =====================
COLORS = {
    "continual": "#2E86AB",      # 蓝色（持续学习）
    "independent": "#A23B72",    # 紫红色（独立学习）
    "joint_per_task": "#F18F01",  # 橙色（联合训练 per-task）
    "joint_mean": "#C73E1D",     # 深红色（联合训练 per-task 均值参考线）
    "nc3_max": "#FF6B6B",         # 红色（持续学习 NC3 Max）
    "nc3_min": "#4ECDC4",         # 青绿色（持续学习 NC3 Min）
}

# ===================== 辅助函数：每5个任务显示一次的刻度标签 =====================
def get_5step_ticks_labels(n_tasks):
    """返回适合横坐标的刻度位置和标签（每5个任务一个标签）"""
    ticks = list(range(0, n_tasks, 5))
    if n_tasks - 1 not in ticks:
        ticks.append(n_tasks - 1)
    labels = [f"T{i}" if i % 5 == 0 else "" for i in ticks]
    if ticks[-1] % 5 != 0:
        labels[-1] = f"T{ticks[-1]}"
    return ticks, labels


def _align(vals, n_tasks):
    """把序列对齐到 n_tasks 长度（不足补边缘，超出截断）"""
    if len(vals) < n_tasks:
        return np.pad(vals, (0, n_tasks - len(vals)), mode="edge")
    return vals[:n_tasks]


def plot_nc3_comparison(result_dir, output_dir):
    """
    绘制 NC3 随任务变化对比曲线（含持续学习 NC3 Max/Min 散点）。

    [P1 修复] NC3 统一使用 per-task 口径（每任务真实 5 类），三条曲线同源可比：
      - Continual : nc_metrics.csv['nc3']（模型训完 task t 后，在 task t 自己的 5 类上）
      - Joint     : joint_per_task_results.csv['nc3']（联合模型在同一批 5 类上；
                    其高位来自槽位碎片化：W_c 收敛到槽心，偏离每个任务的子簇均值）
      - Independent: independent_results.csv['nc3']（单任务可达参考，非界）
    旧版从 joint_results.csv 取的 "Joint (final)" 水平线已删除：
      旧列 "nc3" 实为 per-task 最小值、"nc3_merged_ref" 是另一种标签空间（槽位超类）
      的均值，两者都不是持续 per-task 曲线的上界，画在同一张图里属于跨口径混用。
    """
    independent_path = os.path.join(result_dir, "independent", "independent_results.csv")
    joint_path = os.path.join(result_dir, "joint", "joint_results.csv")
    joint_per_task_path = os.path.join(result_dir, "joint", "joint_per_task_results.csv")
    nc_metrics_path = os.path.join(result_dir, "nc_metrics.csv")
    for p in (independent_path, joint_per_task_path, nc_metrics_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"文件不存在：{p}")

    independent = pd.read_csv(independent_path)
    # 只保留被 NC4 gating 接受的任务，使 Independent 与 Continual/Joint 在同一批任务上比较
    if "accepted" in independent.columns:
        independent = independent[independent["accepted"] == 1].reset_index(drop=True)
    joint_per_task = pd.read_csv(joint_per_task_path)
    continual = pd.read_csv(nc_metrics_path)

    # 横轴统一为 task index；实际 task 数由 per-task 结果决定（gating 后可能 < 20）
    n_tasks = len(joint_per_task)
    task_idx = np.arange(n_tasks)

    # 各曲线 NC3 值（全部 per-task 口径）
    continual_vals = _align(continual["nc3"].values, n_tasks)
    task_nc3_max = _align(continual["nc3_max"].values, n_tasks)
    task_nc3_min = _align(continual["nc3_min"].values, n_tasks)
    independent_vals = _align(independent["nc3"].values, n_tasks)
    joint_per_task_vals = _align(joint_per_task["nc3"].values, n_tasks)

    # 联合 per-task 均值参考线：优先读 joint_results.csv 新列（显式口径），
    # 否则直接由 per-task 曲线求均值（结果相同，不引入 merged 口径）
    joint_mean = None
    if os.path.exists(joint_path):
        joint_final = pd.read_csv(joint_path)
        if "nc3_per_task_mean" in joint_final.columns:
            joint_mean = float(joint_final["nc3_per_task_mean"].iloc[0])
    if joint_mean is None:
        joint_mean = float(np.mean(joint_per_task_vals))

    # 绘图
    fig, ax = plt.subplots(figsize=(9, 6))

    # ----- 持续学习曲线 -----
    ax.plot(task_idx, continual_vals,
            label="Continual (per-task)", color=COLORS["continual"],
            linewidth=2, marker="o", markersize=4)
    # ----- 持续学习 NC3 Max/Min 散点（zorder=10 置于最上层，不被盖住）-----
    ax.scatter(task_idx, task_nc3_max, color=COLORS["nc3_max"], s=100, marker="^",
               zorder=10, label="Continual NC3 Max")
    ax.scatter(task_idx, task_nc3_min, color=COLORS["nc3_min"], s=100, marker="v",
               zorder=10, label="Continual NC3 Min")
    # ----- 独立学习曲线（单任务可达参考） -----
    ax.plot(task_idx, independent_vals,
            label="Independent (per-task, single-task reference)", color=COLORS["independent"],
            linewidth=2, marker="s", markersize=4)
    # ----- 联合训练 per-task 曲线（碎片化下界：W_c 指向槽心） -----
    ax.plot(task_idx, joint_per_task_vals,
            label="Joint (per-task)", color=COLORS["joint_per_task"],
            linewidth=2, linestyle="-", marker="^", markersize=4)
    # ----- 联合训练整体 NC3 水平线（本图口径 per-task 下的整体值 = 30 个任务均值） -----
    ax.axhline(joint_mean, color=COLORS["joint_mean"], linewidth=2, linestyle="--",
               label="Joint overall (per-task mean) = {0:.3f}".format(joint_mean))

    # 纵轴范围自动适配（含所有数据及 Max/Min）
    all_values = np.concatenate([
        continual_vals, independent_vals, joint_per_task_vals,
        task_nc3_max, task_nc3_min, [joint_mean]
    ])
    min_val = np.min(all_values)
    max_val = np.max(all_values)
    margin = (max_val - min_val) * 0.1 if (max_val - min_val) > 0 else 0.1
    ax.set_ylim(bottom=max(min_val - margin, 0), top=max_val + margin)

    # 样式
    ax.set_title("NC3 Evolution Across Tasks (per-task partition)\n"
                 "smaller NC3 = stronger self-duality", fontsize=14, pad=12)
    ax.set_xlabel("Task Index", fontsize=14)
    ax.set_ylabel("NC3", fontsize=14)
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)

    # 横轴刻度：每5个任务显示一次整数标签
    ticks, _ = get_5step_ticks_labels(n_tasks)
    ax.set_xticks(ticks)
    num_labels = [str(t) if t % 5 == 0 else "" for t in ticks]
    if ticks[-1] % 5 != 0:
        num_labels[-1] = str(ticks[-1])
    ax.set_xticklabels(num_labels)
    ax.set_xlim(0, max(task_idx))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "nc3_comparison.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "nc3_comparison.png"), bbox_inches="tight")
    plt.close()
    print("NC3 对比图已保存（per-task 统一口径）")


def main():
    parser = argparse.ArgumentParser(description="NC3 comparison figure")
    parser.add_argument("--result-dir", type=str,
                        default=os.path.join(CURRENT_DIR, "result", "task_continual_learning"),
                        help="实验结果目录（如 .../imagenetBP）")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(CURRENT_DIR, "paper_figures"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("开始生成 NC3 对比图...")
    plot_nc3_comparison(args.result_dir, args.output_dir)
    print(f"图表保存路径：{args.output_dir}/")


if __name__ == "__main__":
    main()
