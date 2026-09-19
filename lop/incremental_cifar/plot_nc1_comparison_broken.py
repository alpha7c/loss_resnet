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
RESULT_DIR = os.path.join(CURRENT_DIR, "result", "task_continual_learning")
OUTPUT_DIR = os.path.join(CURRENT_DIR, "paper_figures")

INDEPENDENT_PATH = os.path.join(RESULT_DIR, "independent", "independent_results.csv")   # 独立训练
JOINT_PATH = os.path.join(RESULT_DIR, "joint", "joint_results.csv")                       # 联合训练 (final)
JOINT_PER_TASK_PATH = os.path.join(RESULT_DIR, "joint", "joint_per_task_results.csv")     # 联合训练 (每任务)
NC_METRICS_PATH = os.path.join(RESULT_DIR, "nc_metrics.csv")                               # 持续训练

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===================== 论文级配色方案（与 permuted_mnist 保持一致） =====================
COLORS = {
    "continual": "#2E86AB",      # 蓝色（持续学习）
    "independent": "#A23B72",    # 紫红色（独立学习）
    "joint_per_task": "#F18F01",  # 橙色（联合训练 per-task）
    "joint_final": "#C73E1D"       # 深红色（联合训练 final）
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


def plot_nc1_comparison_broken():
    """绘制 NC1 随任务变化对比曲线（断轴版：0-1 区间放大刻画，1 以上压缩）"""
    # 1. 读取数据
    for p in (INDEPENDENT_PATH, JOINT_PATH, JOINT_PER_TASK_PATH, NC_METRICS_PATH):
        if not os.path.exists(p):
            raise FileNotFoundError(f"文件不存在：{p}")

    independent = pd.read_csv(INDEPENDENT_PATH)
    joint_final = pd.read_csv(JOINT_PATH)
    joint_per_task = pd.read_csv(JOINT_PER_TASK_PATH)
    continual = pd.read_csv(NC_METRICS_PATH)

    # 2. 横轴统一为 task index
    task_idx = np.arange(20)
    n_tasks = len(task_idx)

    # 3. 各曲线 NC1 值
    continual_vals = _align(continual["nc1"].values, n_tasks)
    independent_vals = _align(independent["nc1"].values, n_tasks)
    joint_per_task_vals = _align(joint_per_task["nc1"].values, n_tasks)
    joint_final_vals = np.full(shape=task_idx.shape,
                               fill_value=joint_final["nc1"].iloc[0], dtype=np.float64)

    # 4. 断轴配置
    y_break = 1.0                                   # 断点：0-1 与 1 以上分界
    top_ymax = float(np.ceil(np.max(continual_vals)))  # 上轴上限（持续训练峰值 5.57 -> 6）

    # 上下两个子图共享横轴：上轴(1 以上, 压缩) / 下轴(0-1, 放大)
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True, figsize=(9, 7),
        gridspec_kw={"height_ratios": [1, 2.5], "hspace": 0.06}
    )

    curves = [
        (continual_vals,      "Continual",        COLORS["continual"],      "o", "-"),
        (independent_vals,    "Independent",      COLORS["independent"],    "s", "-"),
        (joint_per_task_vals, "Joint (per-task)", COLORS["joint_per_task"], "^", "-"),
        (joint_final_vals,    "Joint (final)",    COLORS["joint_final"],    "D", "--"),
    ]

    # 两个轴都画完整曲线（超出各自 ylim 的部分被自动裁剪）
    for ax in (ax_top, ax_bot):
        for vals, label, color, marker, ls in curves:
            ax.plot(task_idx, vals, label=label, color=color,
                    linewidth=2, marker=marker, markersize=4, linestyle=ls)

    # 设置各自 y 范围
    ax_bot.set_ylim(0, y_break)
    ax_top.set_ylim(y_break, top_ymax)

    # 隐藏相邻的脊线
    ax_bot.spines["top"].set_visible(False)
    ax_top.spines["bottom"].set_visible(False)
    ax_top.tick_params(bottom=False, labelbottom=False)  # 上轴不显示横轴刻度

    # 断口斜线标记
    d = 0.015
    kwargs = dict(transform=ax_top.transAxes, color="k", clip_on=False, linewidth=1.2)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)             # 上轴左下斜线
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)       # 上轴右下斜线
    kwargs.update(transform=ax_bot.transAxes)
    ax_bot.plot((-d, +d), (1 - d, 1 + d), **kwargs)       # 下轴左上斜线
    ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs) # 下轴右上斜线

    # 5. 刻度：下轴 0-1 精细刻度，上轴 1 以上稀疏刻度
    ax_bot.set_yticks(np.arange(0.0, 1.0001, 0.2))
    ax_top.set_yticks(np.arange(2.0, top_ymax + 1e-9, 2.0))  # 2, 4, 6

    # 横轴刻度：每5个任务显示一次整数标签
    ticks, _ = get_5step_ticks_labels(n_tasks)
    ax_bot.set_xticks(ticks)
    num_labels = [str(t) if t % 5 == 0 else "" for t in ticks]
    if ticks[-1] % 5 != 0:
        num_labels[-1] = str(ticks[-1])
    ax_bot.set_xticklabels(num_labels)
    ax_bot.set_xlim(0, max(task_idx))

    # 6. 标签与图例
    ax_bot.set_xlabel("Task Index", fontsize=14)
    fig.text(0.02, 0.5, "NC1", va="center", rotation="vertical", fontsize=14)
    ax_bot.legend(fontsize=11, loc="best")
    ax_bot.grid(True, alpha=0.3)
    ax_top.grid(True, alpha=0.3)

    fig.suptitle("NC1 Evolution Across Tasks (0-1 enlarged, >1 compressed)", fontsize=16, y=0.97)

    plt.savefig(os.path.join(OUTPUT_DIR, "nc1_comparison_broken.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(OUTPUT_DIR, "nc1_comparison_broken.png"), bbox_inches="tight")
    plt.close()
    print("NC1 对比图（断轴版）已保存")


def main():
    print("开始生成 NC1 对比图（断轴版）...")
    plot_nc1_comparison_broken()
    print(f"图表保存路径：{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
