import os
import pandas as pd
import matplotlib.pyplot as plt

# ====================== 配置 ======================
NUM_TASKS = 20
BASE_DIR = "1"
SAVE_DIR = "figures"
os.makedirs(SAVE_DIR, exist_ok=True)

# ====================== 读取数据 ======================
def load_all_data():
    # 持续学习
    df_cont = pd.read_csv(os.path.join(BASE_DIR, "0.csv")).sort_values("task_idx")
    # 联合训练
    df_joint = pd.read_csv(os.path.join(BASE_DIR, "joint_results", f"joint_model_per_task_accuracy_{NUM_TASKS}_tasks.csv")).sort_values("task_idx")
    # 独立学习
    df_indep = pd.read_csv(os.path.join(BASE_DIR, "independent_final_results.csv")).sort_values("task_idx")
    return df_cont, df_joint, df_indep

# ====================== 核心：双轴三者关系图 ======================
def plot_task_iso_equi():
    df_cont, df_joint, df_indep = load_all_data()
    x = df_cont["task_idx"]

    plt.figure(figsize=(12, 7))

    # 左 Y：等范数性 Equinormity（实线）
    ax1 = plt.gca()
    ax1.plot(x, df_cont["equinormity"], 'o-', color='#1f77b4', linewidth=2.5, markersize=5, label='Continual | Equinormity')
    ax1.plot(x, df_joint["equinormity"], 's-', color='#ff7f0e', linewidth=2.5, markersize=5, label='Joint | Equinormity')
    ax1.plot(x, df_indep["equinormity"], '^-', color='#2ca02c', linewidth=2.5, markersize=5, label='Independent | Equinormity')
    ax1.set_ylabel("Equinormity (左轴)", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Task Index", fontsize=13, fontweight='bold')

    # 右 Y：等角性 Isotropy（虚线）
    ax2 = ax1.twinx()
    ax2.plot(x, df_cont["isotropy"], 'o--', color='#1f77b4', linewidth=2.5, markersize=5, alpha=0.7, label='Continual | Isotropy')
    ax2.plot(x, df_joint["isotropy"], 's--', color='#ff7f0e', linewidth=2.5, markersize=5, alpha=0.7, label='Joint | Isotropy')
    ax2.plot(x, df_indep["isotropy"], '^--', color='#2ca02c', linewidth=2.5, markersize=5, alpha=0.7, label='Independent | Isotropy')
    ax2.set_ylabel("Isotropy (右轴)", fontsize=13, fontweight='bold')

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

    plt.title("Task ↔ Equinormity ↔ Isotropy 三者关系", fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(SAVE_DIR, "task_iso_equi_dual_axis.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print("✅ 三者关系图已保存：", save_path)

if __name__ == "__main__":
    plot_task_iso_equi()