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

# 准确率矩阵：行为「训练到第 t 个任务」，列为「在第 k 个任务上测试」
ACC_MATRIX_PATH = os.path.join(RESULT_DIR, "task_accuracies", "index-0.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

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


def plot_accuracy_heatmap():
    """绘制持续学习准确率热力图（灾难性遗忘核心图）"""
    if not os.path.exists(ACC_MATRIX_PATH):
        raise FileNotFoundError(f"准确率矩阵文件不存在：{ACC_MATRIX_PATH}")

    acc_matrix_df = pd.read_csv(ACC_MATRIX_PATH, index_col=0)
    acc_matrix = acc_matrix_df.values

    fig, ax = plt.subplots(figsize=(10, 8))
    # 绘制热力图（viridis 配色，0-1 范围）
    im = ax.imshow(acc_matrix, cmap="viridis", vmin=0, vmax=1)
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Accuracy", fontsize=14)

    # 设置坐标轴（适配任务数）
    n_tasks = acc_matrix.shape[0]
    ax.set_xticks(np.arange(n_tasks))
    ax.set_yticks(np.arange(n_tasks))
    # ===== 修改横坐标标签：每5个任务显示一次 =====
    ticks, labels = get_5step_ticks_labels(n_tasks)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticklabels([f"After T{i}" for i in range(n_tasks)])

    # 添加标题和标签
    ax.set_title("Catastrophic Forgetting in Continual Learning", fontsize=16, pad=20)
    ax.set_xlabel("Test Task", fontsize=14)
    ax.set_ylabel("After Training Task", fontsize=14)

    # 优化布局
    plt.tight_layout()
    # 保存（PDF矢量图 + PNG位图）
    plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_heatmap.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_heatmap.png"), bbox_inches="tight")
    plt.close()
    print("准确率热力图已保存")


def main():
    print("开始生成准确率热力图...")
    plot_accuracy_heatmap()
    print(f"图表保存路径：{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
