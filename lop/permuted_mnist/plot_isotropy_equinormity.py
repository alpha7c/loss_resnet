import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------------- 配置参数 ----------------------
BASE_SAVE_DIR = "ceshiewc/"
JOINT_RESULTS_DIR = os.path.join(BASE_SAVE_DIR, "joint_results")
NUM_TASKS = 30

# ---------------------- 读取数据 ----------------------
def read_continual_data():
    path = os.path.join(BASE_SAVE_DIR, "0.csv")
    df = pd.read_csv(path)
    return df["isotropy"].values, df["equinormity"].values, df["task_idx"].values

def read_joint_data():
    path = os.path.join(JOINT_RESULTS_DIR, f"joint_model_per_task_accuracy_{NUM_TASKS}_tasks.csv")
    df = pd.read_csv(path)
    return df["isotropy"].values, df["equinormity"].values, df["task_idx"].values

def read_independent_data():
    path = os.path.join(BASE_SAVE_DIR, "independent_final_results.csv")
    df = pd.read_csv(path)
    return df["isotropy"].values, df["equinormity"].values, df["task_idx"].values

# 读取数据
cont_iso, cont_equi, cont_tasks = read_continual_data()
joint_iso, joint_equi, joint_tasks = read_joint_data()
indep_iso, indep_equi, indep_tasks = read_independent_data()

# ---------------------- 统一坐标轴范围 ----------------------
all_isotropy = np.concatenate([cont_iso, joint_iso, indep_iso])
all_equinormity = np.concatenate([cont_equi, joint_equi, indep_equi])

x_min, x_max = all_isotropy.min(), all_isotropy.max()
y_min, y_max = all_equinormity.min(), all_equinormity.max()

# 增加一点边距
x_margin = (x_max - x_min) * 0.05
y_margin = (y_max - y_min) * 0.05
x_lim = (x_min - x_margin, x_max + x_margin)
y_lim = (y_min - y_margin, y_max + y_margin)

# ---------------------- 绘图：3张子图并排 + 右侧独立颜色条 ----------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 6), dpi=150)
cmap = plt.cm.coolwarm
norm = plt.Normalize(vmin=0, vmax=NUM_TASKS)

# 子图1：持续学习 Continual
ax1 = axes[0]
sc1 = ax1.scatter(cont_iso, cont_equi, c=cont_tasks, cmap=cmap, norm=norm,
                  marker='o', s=70, alpha=0.9)
ax1.set_title("Continual Learning", fontsize=14, fontweight='bold')
ax1.set_xlabel("Isotropy (Equiangularity)", fontsize=12)
ax1.set_ylabel("Equinormity", fontsize=12)
ax1.grid(alpha=0.3, linestyle='--')
ax1.set_xlim(x_lim)
ax1.set_ylim(y_lim)

# 子图2：联合训练 Joint
ax2 = axes[1]
sc2 = ax2.scatter(joint_iso, joint_equi, c=joint_tasks, cmap=cmap, norm=norm,
                  marker='s', s=70, alpha=0.9)
ax2.set_title("Joint Training", fontsize=14, fontweight='bold')
ax2.set_xlabel("Isotropy (Equiangularity)", fontsize=12)
ax2.grid(alpha=0.3, linestyle='--')
ax2.set_xlim(x_lim)
ax2.set_ylim(y_lim)

# 子图3：独立学习 Independent
ax3 = axes[2]
sc3 = ax3.scatter(indep_iso, indep_equi, c=indep_tasks, cmap=cmap, norm=norm,
                  marker='^', s=70, alpha=0.9)
ax3.set_title("Independent Learning", fontsize=14, fontweight='bold')
ax3.set_xlabel("Isotropy (Equiangularity)", fontsize=12)
ax3.grid(alpha=0.3, linestyle='--')
ax3.set_xlim(x_lim)
ax3.set_ylim(y_lim)

# ---------------------- 右侧独立颜色条 ----------------------
# 调整子图布局，留出右侧空间给颜色条
plt.subplots_adjust(right=0.88)
cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
cbar = fig.colorbar(sc1, cax=cbar_ax)
cbar.set_label("Task Index", fontsize=12, rotation=270, labelpad=15)

# 总标题
fig.suptitle("Isotropy vs Equinormity Across Training Paradigms", fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 0.88, 0.96])

# 保存
save_path = os.path.join(BASE_SAVE_DIR, "isotropy_equinormity_3subplots_with_right_cbar.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ 绘图完成！已保存到：{save_path}")