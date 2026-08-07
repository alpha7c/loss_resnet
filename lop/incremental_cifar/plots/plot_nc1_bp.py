"""
NC1 指标可视化 — 仅 BP 算法，三种训练范式对比
=================================================
读取持续学习、联合训练、独立学习的 NC1 数据，绘制跨任务折线图。

输入文件（均位于 results/task_continual_learning/ 下）：
  - nc_metrics.csv                    持续学习 per-task NC1
  - joint/joint_per_task_results.csv   联合训练 per-task NC1
  - independent/independent_results.csv  独立学习 per-task NC1

输出：plots/nc1_bp_comparison.png / pdf

用法：直接运行 python plot_nc1_bp.py
"""

import os
import re
import csv
import numpy as np
import matplotlib.pyplot as plt

# ===================== 路径配置 =====================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CURRENT_DIR, "..", "results", "task_continual_learning")
OUTPUT_DIR = CURRENT_DIR  # 输出到 plots/ 目录

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===================== 论文级绘图风格 =====================
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.dpi"] = 150

# ===================== 配色方案 =====================
COLORS = {
    "continual": "#2E86AB",     # 蓝色（持续学习）
    "independent": "#A23B72",   # 紫红色（独立学习）
    "joint": "#F18F01",         # 橙色（联合训练）
}


# ===================== 辅助函数 =====================

def _parse_tensor_str(val):
    """
    解析 CSV 中可能出现的 tensor(...) 字符串，提取纯数值。
    """
    if isinstance(val, (int, float, np.floating, np.integer)):
        return float(val)
    s = str(val).strip()
    try:
        return float(s)
    except ValueError:
        pass
    m = re.search(r'tensor\(([\d.e+\-]+)', s)
    if m:
        return float(m.group(1))
    raise ValueError(f"无法解析 NC1 值: '{val}'")


def _read_csv_col(path, col_name):
    """读取 CSV 中指定列，跳过表头，返回 numpy 数组"""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(_parse_tensor_str(row[col_name]))
    return np.array(rows)


def read_continual_nc1():
    """读取持续学习 per-task NC1"""
    path = os.path.join(RESULTS_DIR, "nc_metrics.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    nc1 = _read_csv_col(path, "nc1")
    # task 列名是 "task"
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        tasks = np.array([int(row["task"]) for row in reader])
    return tasks, nc1


def read_joint_nc1():
    """读取联合训练 per-task NC1"""
    path = os.path.join(RESULTS_DIR, "joint", "joint_per_task_results.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    nc1 = _read_csv_col(path, "nc1")
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        tasks = np.array([int(row["task_idx"]) for row in reader])
    return tasks, nc1


def read_independent_nc1():
    """读取独立学习 per-task NC1"""
    path = os.path.join(RESULTS_DIR, "independent", "independent_results.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    nc1 = _read_csv_col(path, "nc1")
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        tasks = np.array([int(row["task_idx"]) for row in reader])
    return tasks, nc1


# ===================== 绘图函数 =====================

def plot_nc1_comparison():
    """绘制 NC1 三种范式对比折线图"""

    # 1. 读取数据
    cont_tasks, cont_nc1 = read_continual_nc1()
    joint_tasks, joint_nc1 = read_joint_nc1()
    indep_tasks, indep_nc1 = read_independent_nc1()

    # 过滤掉零值（NC 计算失败的 task）
    cont_valid = cont_nc1 > 1e-6
    cont_t_clean = cont_tasks[cont_valid]
    cont_n_clean = cont_nc1[cont_valid]

    print(f"持续学习: {len(cont_nc1)} tasks → 有效 {len(cont_n_clean)} 个")
    for t, v in zip(cont_tasks, cont_nc1):
        tag = " ✓" if v > 1e-6 else " ✗ (0)"
        print(f"  Task {t}: NC1 = {v:.4f}{tag}")
    print(f"联合训练: {len(joint_nc1)} tasks")
    for t, v in zip(joint_tasks, joint_nc1):
        print(f"  Task {t}: NC1 = {v:.4f}")
    print(f"独立学习: {len(indep_nc1)} tasks")
    for t, v in zip(indep_tasks, indep_nc1):
        print(f"  Task {t}: NC1 = {v:.4f}")

    n_tasks = max(max(cont_tasks), max(joint_tasks), max(indep_tasks)) + 1

    # 2. 创建画布
    fig, ax = plt.subplots(figsize=(10, 6))

    # 3. 绘制三条曲线
    ax.plot(cont_t_clean, cont_n_clean,
            label="Continual (BP)", color=COLORS["continual"],
            linewidth=2, marker="o", markersize=7, zorder=3)

    ax.plot(joint_tasks, joint_nc1,
            label="Joint (BP)", color=COLORS["joint"],
            linewidth=2, marker="s", markersize=7, zorder=3)

    ax.plot(indep_tasks, indep_nc1,
            label="Independent (BP)", color=COLORS["independent"],
            linewidth=2, marker="^", markersize=7, zorder=3)

    # 4. 平均值水平虚线
    for vals, color, name in [
        (cont_n_clean, COLORS["continual"], "Continual"),
        (joint_nc1, COLORS["joint"], "Joint"),
        (indep_nc1, COLORS["independent"], "Independent"),
    ]:
        if len(vals) > 0:
            mean_val = np.mean(vals)
            ax.axhline(y=mean_val, color=color, linestyle="--",
                       linewidth=1.2, alpha=0.6,
                       label=f"{name} Mean = {mean_val:.3f}")

    # 5. 坐标轴
    ax.set_title("NC1 Across Tasks — BP Algorithm (CIFAR-100, ResNet18)",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Task Index", fontsize=13)
    ax.set_ylabel("NC1", fontsize=13)
    ax.set_xticks(range(n_tasks))

    # 6. 纵轴自适应
    all_vals = np.concatenate([cont_n_clean, joint_nc1, indep_nc1])
    y_min = max(0, np.min(all_vals) * 0.9)
    y_max = np.max(all_vals) * 1.15
    ax.set_ylim(y_min, y_max)

    # 7. 图例和网格
    ax.legend(fontsize=10, loc="upper left",
              framealpha=0.9, edgecolor="gray")
    ax.grid(True, alpha=0.3, linestyle="--")

    # 8. 保存
    plt.tight_layout()
    out_pdf = os.path.join(OUTPUT_DIR, "nc1_bp_comparison.pdf")
    out_png = os.path.join(OUTPUT_DIR, "nc1_bp_comparison.png")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"\n✅ NC1 对比图已保存:")
    print(f"   {out_pdf}")
    print(f"   {out_png}")


# ===================== 主函数 =====================

def main():
    print("=" * 60)
    print("  NC1 可视化 — BP 算法 · 三种训练范式")
    print("=" * 60)
    plot_nc1_comparison()
    print("\n完成。")


if __name__ == "__main__":
    main()
