import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===================== 修复字体警告 + 论文级绘图风格 =====================
plt.rcParams["font.family"] = "DejaVu Sans"  # 替换Times New Roman为服务器存在的字体
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
    "joint_final": "#C73E1D"       # 深红色（联合训练 merged/slot）
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


def _joint_col(joint_final, new_name, old_name):
    """兼容 joint_results.csv 新旧列名（新列显式带 partition 后缀，旧列名有口径混用问题）"""
    if new_name in joint_final.columns:
        return float(joint_final[new_name].iloc[0])
    if old_name in joint_final.columns:
        return float(joint_final[old_name].iloc[0])
    raise KeyError("joint_results.csv 缺少列 {0}（或旧列名 {1}）".format(new_name, old_name))


def _style_axes(ax, task_idx, n_tasks):
    ticks, _ = get_5step_ticks_labels(n_tasks)
    ax.set_xticks(ticks)
    num_labels = [str(t) if t % 5 == 0 else "" for t in ticks]
    if ticks[-1] % 5 != 0:
        num_labels[-1] = str(ticks[-1])
    ax.set_xticklabels(num_labels)
    if len(task_idx) > 1:
        ax.set_xlim(task_idx[0], task_idx[-1])
    ax.set_xlabel("Task Index", fontsize=14)
    ax.grid(True, alpha=0.3)


def plot_nc1_slot_partition(result_dir, output_dir, slot_path, joint_final):
    """
    [P1 修复后的主模式] NC1 统一在 slot-partition（累积已见任务、槽位标签 0-4）上比较：
      - Continual (slot)：eval_slot_partition_nc.py 离线算出的 nc1_slot 曲线
      - Joint (slot)：--include-joint 时同子集评估的 joint_nc1_slot 曲线；
                      否则退化为 joint_results.csv 的 nc1_merged 水平线（同口径上确界）
    两条曲线标签空间一致（Σ_W 都含槽内子簇散布），数值才可比。
    理论上持续侧随任务数单调上升、趋近联合侧。
    """
    slot = pd.read_csv(slot_path)
    n_tasks = len(slot)
    task_idx = slot["task_idx"].values
    continual_vals = slot["nc1_slot"].values

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(task_idx, continual_vals,
            label="Continual (slot, accumulated)", color=COLORS["continual"],
            linewidth=2, marker="o", markersize=4)

    all_values = [continual_vals]
    # 联合训练整体 NC1：joint 模型在原生 slot 标签空间（全部任务）上的值，
    # 同时也是持续侧 slot 曲线的理论上界（Σ_W 含槽内全部子簇散布）
    merged = _joint_col(joint_final, "nc1_merged", "nc1")
    ax.axhline(merged, color=COLORS["joint_final"], linewidth=2, linestyle="--",
               label="Joint overall (slot partition) = {0:.3f}".format(merged))
    all_values.append(np.array([merged]))
    if "joint_nc1_slot" in slot.columns:
        joint_vals = slot["joint_nc1_slot"].values
        ax.plot(task_idx, joint_vals,
                label="Joint (slot, same subset)", color=COLORS["joint_per_task"],
                linewidth=2, linestyle="-", marker="D", markersize=4)
        all_values.append(joint_vals)

    vals = np.concatenate(all_values)
    margin = (vals.max() - vals.min()) * 0.1 if vals.max() > vals.min() else 0.1
    ax.set_ylim(bottom=max(vals.min() - margin, 0), top=vals.max() + margin)
    ax.set_title("NC1 Evolution Across Tasks (slot partition, accumulated seen tasks)\n"
                 "smaller NC1 = stronger collapse", fontsize=14, pad=12)
    ax.set_ylabel("NC1", fontsize=14)
    ax.legend(fontsize=11, loc="best")
    _style_axes(ax, task_idx, n_tasks)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "nc1_comparison.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "nc1_comparison.png"), bbox_inches="tight")
    plt.close()
    print("NC1 对比图（slot-partition 口径）已保存")


def plot_nc1_per_task(result_dir, output_dir, joint_per_task, joint_final):
    """
    [回退模式] slot-partition 评估尚未运行时，退回 per-task 口径——
    但三条曲线必须全部来自 per-task 源（P1 修复点：绝不再把 merged 水平线
    与 per-task 曲线画进同一张图）。
    注意：per-task 口径下 NC1 对"模型是谁"不敏感（K=5、Σ_B 秩 4），
    联合与持续基本重合，此图仅用于检查数据，不用于论文结论。
    """
    nc_metrics_path = os.path.join(result_dir, "nc_metrics.csv")
    independent_path = os.path.join(result_dir, "independent", "independent_results.csv")
    for p in (nc_metrics_path, independent_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"文件不存在：{p}")

    continual = pd.read_csv(nc_metrics_path)
    independent = pd.read_csv(independent_path)
    if "accepted" in independent.columns:
        independent = independent[independent["accepted"] == 1].reset_index(drop=True)

    n_tasks = len(joint_per_task)
    task_idx = np.arange(n_tasks)
    continual_vals = _align(continual["nc1"].values, n_tasks)
    independent_vals = _align(independent["nc1"].values, n_tasks)
    joint_per_task_vals = _align(joint_per_task["nc1"].values, n_tasks)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(task_idx, continual_vals,
            label="Continual (per-task)", color=COLORS["continual"],
            linewidth=2, marker="o", markersize=4)
    ax.plot(task_idx, independent_vals,
            label="Independent (per-task, single-task reference)", color=COLORS["independent"],
            linewidth=2, marker="s", markersize=4)
    ax.plot(task_idx, joint_per_task_vals,
            label="Joint (per-task)", color=COLORS["joint_per_task"],
            linewidth=2, linestyle="-", marker="^", markersize=4)
    # 联合训练整体 NC1 水平线：joint 模型在原生 slot 标签空间（全部任务）上的值。
    # 注意口径：曲线是 per-task，此线是 slot-partition，图例中已显式标注；
    # 它同时是持续侧 slot 口径曲线（eval_slot_partition_nc.py）的理论收敛上界。
    joint_overall = _joint_col(joint_final, "nc1_merged", "nc1")
    ax.axhline(joint_overall, color=COLORS["joint_final"], linewidth=2, linestyle="--",
               label="Joint overall (slot partition) = {0:.3f}".format(joint_overall))

    all_values = np.concatenate([continual_vals, independent_vals, joint_per_task_vals,
                                 [joint_overall]])
    min_val, max_val = np.min(all_values), np.max(all_values)
    margin = (max_val - min_val) * 0.1 if (max_val - min_val) > 0 else 0.1
    ax.set_ylim(bottom=max(min_val - margin, 0), top=max_val + margin)
    ax.set_title("NC1 Evolution Across Tasks (per-task partition)\n"
                 "smaller NC1 = stronger collapse", fontsize=14, pad=12)
    ax.set_ylabel("NC1", fontsize=14)
    ax.legend(fontsize=10, loc="best")
    _style_axes(ax, task_idx, n_tasks)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "nc1_comparison.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "nc1_comparison.png"), bbox_inches="tight")
    plt.close()
    print("NC1 对比图（per-task 回退口径）已保存")


def plot_nc1_comparison(result_dir, output_dir):
    joint_path = os.path.join(result_dir, "joint", "joint_results.csv")
    joint_per_task_path = os.path.join(result_dir, "joint", "joint_per_task_results.csv")
    slot_path = os.path.join(result_dir, "continual_slot_partition_nc.csv")
    for p in (joint_path, joint_per_task_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"联合训练结果文件不存在：{p}")

    joint_final = pd.read_csv(joint_path)
    joint_per_task = pd.read_csv(joint_per_task_path)

    if os.path.exists(slot_path):
        plot_nc1_slot_partition(result_dir, output_dir, slot_path, joint_final)
    else:
        print("[warn] 未找到 continual_slot_partition_nc.csv，"
              "回退到 per-task 口径（先运行 eval_slot_partition_nc.py 可获得可比的 slot 口径）")
        plot_nc1_per_task(result_dir, output_dir, joint_per_task, joint_final)


def main():
    parser = argparse.ArgumentParser(description="NC1 comparison figure")
    parser.add_argument("--result-dir", type=str,
                        default=os.path.join(CURRENT_DIR, "result", "task_continual_learning"),
                        help="实验结果目录（如 .../imagenetBP）")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(CURRENT_DIR, "paper_figures"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("开始生成 NC1 对比图...")
    plot_nc1_comparison(args.result_dir, args.output_dir)
    print(f"图表保存路径：{args.output_dir}/")


if __name__ == "__main__":
    main()
