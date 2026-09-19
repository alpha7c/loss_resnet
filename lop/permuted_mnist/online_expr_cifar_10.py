# ====================== NC3 t-SNE 可视化独立模块（完整版） ======================
import matplotlib.pyplot as plt
import os
import gc
import numpy as np
from sklearn.manifold import TSNE
from lop.algos.ewc import EWC

def visualize_nc3_tsne(class_centers, classifier_weights, task_idx, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 原始数据
    cc = class_centers.detach().cpu().numpy()          # [10, D]  类中心
    clf = classifier_weights.detach().cpu().numpy()   # [10, D]  分类器权重
    
    # 2. 计算平均值
    cc_mean = cc.mean(axis=0, keepdims=True)          # [1, D]  类中心均值
    clf_mean = clf.mean(axis=0, keepdims=True)        # [1, D]  权重均值
    
    # 3. 拼接所有需要降维的向量
    # 顺序：10个类中心 + 10个权重 + 类中心均值 + 权重均值
    all_vectors = np.concatenate([cc, clf, cc_mean, clf_mean], axis=0)
    
    # 4. t-SNE 降维
    tsne = TSNE(
        n_components=2,
        perplexity=5,
        random_state=42,
        init="random"
    )
    vec_2d = tsne.fit_transform(all_vectors)
    
    # 5. 拆分降维后的数据
    centers_2d = vec_2d[:10]          # 0~9: 类中心
    clf_2d     = vec_2d[10:20]         # 10~19: 分类器权重
    cc_mean_2d = vec_2d[20:21]         # 20: 类中心平均
    clf_mean_2d = vec_2d[21:22]        # 21: 权重平均
    
    # 6. 绘图
    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # ----------------------
    # 画 10 个类中心（圆圈）
    # ----------------------
    for i in range(10):
        plt.scatter(
            centers_2d[i, 0], centers_2d[i, 1],
            color=colors[i], s=180, marker='o',
            edgecolors='black', linewidth=2,
            label=f"Class {i} Center" if i == 0 else ""
        )
    
    # ----------------------
    # 画 10 个分类器权重（方块）
    # ----------------------
    for i in range(10):
        plt.scatter(
            clf_2d[i, 0], clf_2d[i, 1],
            color=colors[i], s=180, marker='s',
            edgecolors='black', linewidth=2,
            label=f"Class {i} Weight" if i == 0 else ""
        )
    
    # ----------------------
    # 画一一对应连线（中心 ↔ 权重）
    # ----------------------
    for i in range(10):
        plt.plot(
            [centers_2d[i, 0], clf_2d[i, 0]],
            [centers_2d[i, 1], clf_2d[i, 1]],
            color=colors[i], alpha=0.6, linewidth=2, linestyle='-'
        )
    
    # ----------------------
    # 画平均值（大红星 + 大蓝星）
    # ----------------------
    plt.scatter(
        cc_mean_2d[0, 0], cc_mean_2d[0, 1],
        color='red', marker='*', s=500,
        edgecolors='black', linewidth=3,
        label="Mean Class Center"
    )
    plt.scatter(
        clf_mean_2d[0, 0], clf_mean_2d[0, 1],
        color='blue', marker='*', s=500,
        edgecolors='black', linewidth=3,
        label="Mean Classifier Weight"
    )
    
    # ----------------------
    # 画平均值连线
    # ----------------------
    plt.plot(
        [cc_mean_2d[0, 0], clf_mean_2d[0, 0]],
        [cc_mean_2d[0, 1], clf_mean_2d[0, 1]],
        color='black', alpha=0.8, linewidth=3, linestyle='--',
        label="Mean ↔ Mean"
    )
    
    # ----------------------
    # 图表样式
    # ----------------------
    plt.title(f"Task {task_idx} NC3 Visualization (t-SNE)", fontsize=16)
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.legend(fontsize=10, loc='best')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # 保存
    save_path = os.path.join(save_dir, f"task_{task_idx}_nc3_tsne.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ NC3 完整版可视化已保存：{save_path}")


    #改显存
    plt.close('all')   # 关闭所有图
    del vec_2d, all_vectors, cc, clf, cc_mean, clf_mean
    gc.collect()
# ====================== 模块结束 ======================


# ====================== 新增：统一生成所有任务的排列 ======================
def generate_all_task_permutations(num_tasks, input_size, num_samples):
    """
    统一生成所有任务的像素排列 + 数据排列
    返回：tasks_permutations = [(pixel_perm0, data_perm0), (pixel_perm1, data_perm1), ...]
    """
    tasks_permutations = []
    for _ in range(num_tasks):
        pixel_perm = np.random.permutation(input_size)
        data_perm = np.random.permutation(num_samples)
        tasks_permutations.append((pixel_perm, data_perm))
    return tasks_permutations
# ====================== 新增结束 ======================





import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 使用第一张GPU

import sys
import json
import torch
import torch.nn.functional as F
print("=== 调试：GPU 可见性 ===")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU 数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"当前可见设备: {os.environ.get('CUDA_VISIBLE_DEVICES', '未设置')}")
import argparse
import pickle
import gc
import numpy as np
from tqdm import tqdm
from lop.algos.bp import Backprop
from lop.algos.cbp import ContinualBackprop
from lop.nets.linear import MyLinear
from torch.nn.functional import softmax
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from lop.nets.deep_ffnn import DeepFFNN
from lop.utils.miscellaneous import nll_accuracy, compute_matrix_rank_summaries
from lop.utils.neural_collapse import NC
from cifar10_data import load_cifar10
from mnist_data import load_mnist
# ===== ResNet 迁移：复用 incremental_cifar 的模型与 learner 栈 =====
from lop.nets.torchvision_modified_resnet import build_resnet18, kaiming_init_resnet_module
from lop.incremental_cifar.learners import build_learner

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Optimize CUDA memory allocation (reduce fragmentation)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

plt.switch_backend('Agg')  

dead_neuron_threshold = 1e-8

NUM_EPOCHS = 50

# ===== ResNet 迁移全局开关（在 online_expr() 中按配置 net_type 设置）=====
NET_TYPE = 'mlp'            # 'mlp' (DeepFFNN, 原逻辑) 或 'resnet' (ResNet-18)
RESHAPE_TO_4D = False       # True 时 generate_task_samples 将排列后的扁平向量 reshape 为 [N,C,H,W]
INPUT_CHANNELS = 3          # ResNet 输入通道数（CIFAR-10=3, MNIST=1）
INPUT_H = 32                # ResNet 输入高度（CIFAR-10=32, MNIST=28）
INPUT_W = 32                # ResNet 输入宽度（CIFAR-10=32, MNIST=28）
EXAMPLES_PER_TASK = 50000   # 每个任务的样本数（CIFAR-10=50000, MNIST=60000）
EVAL_BATCH_SIZE = 1000      # 评估/前向测试 batch size（ResNet 需调小防 OOM）
TASK_ACC_THRESHOLD = 0.39   # 单任务/联合训练提前停止准确率阈值
NC1_INTERVAL = 60000        # 训练中 NC 记录间隔（按 mini-batch 迭代数计，batch 变大时需相应调小）


import copy
import math

def compute_avg_loss(model, dataloader, device, loss_func=F.cross_entropy):
    """
    计算模型在整个数据集上的平均损失（不更新梯度）。
    使用与 Backprop 相同的 loss_func (F.cross_entropy)。
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)                # 模型输出 logits
            loss = loss_func(logits, y, reduction='sum')
            total_loss += loss.item()
            total_samples += y.size(0)
    model.train()
    return total_loss / total_samples

def compute_param_diff_norm(state_dict_before, state_dict_after):
    """
    计算两个参数字典之间的总 L2 范数差，以及各层的 L2 范数。
    返回 (total_norm, layer_norms_dict)

    注意：ResNet 的 state_dict 中包含 BatchNorm 的 buffer
    (running_mean / running_var / num_batches_tracked)。它们不是梯度
    训练的参数（num_batches_tracked 还是 int64 计数器，平方后会严重
    虚高 Δθ），计算可塑性指标 P_n 时必须排除，否则与 MLP 结果不可比。
    """
    _BN_BUFFER_KEYS = ('running_mean', 'running_var', 'num_batches_tracked')
    total_norm_sq = 0.0
    layer_norms = {}
    for key in state_dict_before.keys():
        if any(bk in key for bk in _BN_BUFFER_KEYS):
            continue  # 跳过 BN buffer，只统计可训练参数
        diff = state_dict_after[key] - state_dict_before[key]
        layer_norm_sq = torch.sum(diff ** 2).item()
        layer_norms[key] = math.sqrt(layer_norm_sq)
        total_norm_sq += layer_norm_sq
    total_norm = math.sqrt(total_norm_sq)
    return total_norm, layer_norms

def train_one_epoch(learner, x_epoch, y_epoch, mini_batch_size, change_after, examples_per_task, dev):
    """
    独立训练一个 epoch，返回更新后的迭代计数器（全局 iter 需要在此函数外定义）。
    注意：该函数会修改全局变量 iter_count，请根据您的代码结构调整。
    """
    global iter_count   # 如果 iter_count 是全局变量，需要声明。或者作为参数传入并返回。
    epoch_permutation = np.random.permutation(examples_per_task)
    x_shuffled = x_epoch[epoch_permutation]
    y_shuffled = y_epoch[epoch_permutation]
    for start_idx in range(0, change_after, mini_batch_size):
        batch_x = x_shuffled[start_idx: start_idx + mini_batch_size]
        batch_y = y_shuffled[start_idx: start_idx + mini_batch_size]
        # 位置参数调用：同时兼容旧 Backprop.learn(x, target) 与新 BPLearner.learn(images, labels)
        learner.learn(batch_x.to(dev), batch_y.to(dev))
        iter_count += 1
    return iter_count

# ====================== 新增：通用死亡神经元计算函数 ======================
def calculate_dead_neurons(learner, x_task, num_hidden_layers, dev):
    """通用函数：计算当前模型的死亡神经元数量（训练后调用）"""
    # ResNet 等卷积网络没有 predict() 接口，且"逐神经元死亡"定义对
    # 4D 卷积特征图不成立，返回 0 占位（保持 CSV 列数不变）
    if not hasattr(learner.net, 'predict'):
        return [0] * num_hidden_layers
    learner.net.eval()  # 评估模式，避免BatchNorm等影响
    with torch.no_grad():
        m = learner.net.predict(x_task[:20000])[1]  # 用前20000样本，和原逻辑一致
        dead_neurons_list = []
        for rep_layer_idx in range(num_hidden_layers):
            neuron_activation_sums = m[rep_layer_idx].abs().sum(dim=0)
            dead = (neuron_activation_sums < dead_neuron_threshold).sum()
            dead_neurons_list.append(int(dead.item()))
        del m
    learner.net.train()  # 切回训练模式
    return dead_neurons_list
# ====================== 新增结束 ======================

# class TrajectoryMap:
#     def __init__(self, model, step=1, save_dir="figures"):
#         self.model = model
#         self.step = step
#         self.save_dir = save_dir
#         self.trajectory = []  
#         os.makedirs(self.save_dir, exist_ok=True) 




    # def record_trajectory(self, current_step, is_train=True):
    #     if not is_train or (current_step % self.step != 0):
    #         return
        
    #     with torch.no_grad():
    #         params_list = []
    #         for param in self.model.parameters():
    #             flat_param = param.data.cpu().flatten()
    #             params_list.append(flat_param)
    #             del flat_param
        
    #     if not params_list:
    #         return
        
    #     with torch.no_grad():
    #         flat_params = torch.cat(params_list)
    #         param_norm = flat_params.norm(2)  
            
    #         eps = 1e-12 
    #         if param_norm < eps:
    #             norm_params = flat_params / (param_norm + eps)
    #         else:
    #             norm_params = flat_params / param_norm  
            
    #         del flat_params, param_norm
        
    #     assert torch.isclose(norm_params.norm(2), torch.tensor(1.0), atol=1e-4), \
    #         f"Parameter normalization failed! Vector length = {norm_params.norm(2)}, which should be close to 1 as required by the paper"
        
    #     self.trajectory.append(norm_params)
    #     del norm_params

    # def compute_trajectory_metrics(self):
    #     if len(self.trajectory) < 2:
    #         return {
    #             "traj_avg_cos": 0.0,    
    #             "traj_min_cos": 0.0,    
    #             "traj_max_cos": 0.0,   
    #             "traj_length": len(self.trajectory)  
    #         }
        
    #     with torch.no_grad():
    #         traj_tensor = torch.stack(self.trajectory)
    #         cos_sim_matrix = traj_tensor @ traj_tensor.T  
    #         cos_sim_matrix = torch.clamp(cos_sim_matrix, min=-1.0, max=1.0)
    #         upper_triangle = cos_sim_matrix.triu(diagonal=1)  
    #         valid_cos_vals = upper_triangle[upper_triangle != 0]  
            
    #         del traj_tensor, cos_sim_matrix, upper_triangle
        
    #     return {
    #         "traj_avg_cos": float(valid_cos_vals.mean().item()),  
    #         "traj_min_cos": float(valid_cos_vals.min().item()),  
    #         "traj_max_cos": float(valid_cos_vals.max().item()),  
    #         "traj_length": len(self.trajectory)                  
    #     }

    # def plot_trajectory_map(self, task_idx, total_iterations):
    #     if len(self.trajectory) == 0:
    #         print(f"[Task {task_idx}] No parameter trajectory recorded, skipping plotting")
    #         return
        
    #     traj_tensor = torch.stack(self.trajectory)
        
    #     cos_sim_matrix = traj_tensor @ traj_tensor.T
        
    #     cos_sim_matrix = torch.clamp(cos_sim_matrix, min=-1.0, max=1.0)
        
    #     plt.figure(figsize=(10, 8))
       
    #     colors = [(1, 1, 0), (0, 1, 0), (0, 0, 1)]
    #     custom_cmap = mcolors.LinearSegmentedColormap.from_list("blu_green_yellow", colors, N=256)
        
    #     im = plt.imshow(cos_sim_matrix.numpy(), cmap=custom_cmap, interpolation="nearest")
    #     plt.colorbar(im, label="Cosine Similarity ")  
    #     plt.title(f"Task {task_idx} Trajectory Map (Total Iterations: {total_iterations})")
        
    #     total_records = len(traj_tensor)
    #     if total_records > 1:
    #         tick_positions = np.linspace(0, total_records - 1, min(21, total_records + 1), dtype=int)
    #         tick_labels = [f"{int(total_iterations * pos / (total_records - 1))}" for pos in tick_positions]
    #     else:
    #         tick_positions = [0]
    #         tick_labels = [f"{0}"]
        
    #     plt.xticks(tick_positions, tick_labels, rotation=45)
    #     plt.yticks(tick_positions, tick_labels)
    #     plt.tight_layout()  
        
    #     save_path = os.path.join(self.save_dir, f"task_{task_idx}_trajectory_map.png")
    #     plt.savefig(save_path, dpi=150)
    #     plt.close()
    #     print(f"[Task {task_idx}] Trajectory Map saved to: {save_path}")


def save_model_params_to_csv(model, save_dir, num_tasks):
    # ===== ResNet：没有 in_layer/layers.* 等 MLP 层名 =====
    # 逐参数展开写 CSV 会产生千万行级文件（ResNet-18 约 11M 参数），
    # 完整参数已由 joint_model_*.pth (state_dict) 保存，此处跳过。
    if not hasattr(model, 'in_layer'):
        print(f"[Joint Training] Non-MLP model ({type(model).__name__}) detected: "
              f"skipping per-layer CSV export (full state_dict already saved as .pth)")
        return

    layer_groups = [
        {
            "layer_keys": ["in_layer.fc.weight", "in_layer.fc.bias"],
            "filename": f'joint_model_params_{num_tasks}_tasks_layer_0_input.csv'
        },
        {
            "layer_keys": ["layers.0.weight", "layers.0.bias"],
            "filename": f'joint_model_params_{num_tasks}_tasks_layer_1_hidden1.csv'
        },
        {
            "layer_keys": ["layers.2.weight", "layers.2.bias"],
            "filename": f'joint_model_params_{num_tasks}_tasks_layer_2_hidden2.csv'
        },
        {
            "layer_keys": ["layers.4.weight", "layers.4.bias"],
            "filename": f'joint_model_params_{num_tasks}_tasks_layer_3_hidden3.csv'
        },
        {
            "layer_keys": ["layers.6.weight", "layers.6.bias"],
            "filename": f'joint_model_params_{num_tasks}_tasks_layer_4_output.csv'
        }
    ]
    
    file_handles = {}
    for group in layer_groups:
        csv_path = os.path.join(save_dir, group["filename"])
        fh = open(csv_path, 'w', encoding='utf-8')
        fh.write('layer_name,param_flat_index,param_value,num_tasks\n')
        file_handles[group["filename"]] = fh
    
    for layer_name, param in model.named_parameters():
        target_group = None
        for group in layer_groups:
            if layer_name in group["layer_keys"]:
                target_group = group
                break
        if not target_group:
            continue  
        
        flat_param = param.data.cpu().flatten()
        fh = file_handles[target_group["filename"]]
        for idx, val in enumerate(flat_param):
            fh.write(f"{layer_name},{idx},{val.item():.10f},{num_tasks}\n")
    
    for fh in file_handles.values():
        fh.close()
    
    print(f"[Joint Training] Model parameters split into 5 CSV files in: {save_dir}")
    for group in layer_groups:
        csv_path = os.path.join(save_dir, group["filename"])
        print(f"  - {csv_path}")


def generate_task_samples(x_original, y_original, pixel_perm, data_perm, examples_per_task, dev):
    x_task = x_original[:, pixel_perm].clone()
    x_task, y_task = x_task[data_perm], y_original[data_perm].clone()

    x_task = x_task[:examples_per_task].to(dev)
    y_task = y_task[:examples_per_task].to(dev)

    # ===== ResNet：像素排列在扁平空间完成后，reshape 回 4D 图像 =====
    # 排列破坏了空间结构，但每个任务的输入分布仍然不同，
    # permuted-task 的实验语义得以保留（卷积先验失效属于预期）。
    if RESHAPE_TO_4D and x_task.dim() == 2:
        x_task = x_task.view(-1, INPUT_CHANNELS, INPUT_H, INPUT_W)
    return x_task, y_task


def release_tensor_memory(*tensors):
    for tensor in tensors:
        if tensor is not None:
            del tensor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ====================== 新增：前向测试函数 ======================
def forward_test(learner, tasks_permutations, x_original, y_original, current_task_idx, save_dir, dev):
    """
    前向测试：在第current_task_idx个任务训练完成后，测试当前网络对所有历史任务(T1~Tcurrent_task_idx)的性能
    """
    learner.net.eval()
    # 前向测试结果保存文件
    forward_test_file = os.path.join(save_dir, 'continual_forward_test_results.csv')
    
    # 初始化文件头（仅第一次创建时）
    if not os.path.exists(forward_test_file):
        headers = [
            'current_train_task_idx',  # 当前训练完成的任务ID
            'test_task_idx',           # 被测试的历史任务ID
            'accuracy',                # 该历史任务的测试准确率
            'nc1', 'nc2', 'nc3', 'nc4', 'isotropy', 'equinormity',# 该历史任务的神经坍缩指标
            'num_samples'              # 该任务的测试样本数
        ]
        with open(forward_test_file, 'w', encoding='utf-8') as f:
            f.write(','.join(headers) + '\n')
    
    examples_per_task = EXAMPLES_PER_TASK
    print(f"\n=== Forward Test: Evaluate current model on all historical tasks (T1~T{current_task_idx}) ===")
    
    # 遍历所有历史任务（0 ~ current_task_idx）
    for test_task_idx in range(current_task_idx + 1):
        print(f"  Testing historical task {test_task_idx}...")
        # 获取该历史任务的像素/数据排列
        pixel_perm, data_perm = tasks_permutations[test_task_idx]
        # 生成该任务的测试样本
        x_task, y_task = generate_task_samples(
            x_original, y_original, pixel_perm, data_perm, examples_per_task, dev
        )
        task_dataset = TensorDataset(x_task, y_task)
        task_dataloader = DataLoader(task_dataset, batch_size=EVAL_BATCH_SIZE)

        # 计算准确率
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for val_x, val_y in task_dataloader:
                val_output = learner.net(val_x)
                preds = torch.argmax(val_output, dim=1)
                total_correct += (preds == val_y).sum().item()
                total_samples += val_y.size(0)
        accuracy = total_correct / total_samples
        
        # 计算NC指标
        #nc1, nc2, nc3, nc4 = NC(model=learner.net, data_loader=task_dataloader, num_classes=10)
        #nc1, nc2, nc3, nc3_max, nc3_min, nc4, isotropy, equinormity = NC(model=learner.net, data_loader=task_dataloader, num_classes=10)
        # 正确的接收方式（6个值）
        nc1, nc2, nc3, nc3_max, nc3_min, nc4, isotropy, equinormity = NC(model=learner.net, data_loader=task_dataloader, num_classes=10)
        # 改显存
        torch.cuda.empty_cache()
        gc.collect()
        # del all_feats, all_labels, class_centers, classifier_weight  # 把 NC 里的大张量都删掉
        # torch.cuda.empty_cache()
        # gc.collect()
        #learner.net.train()  # 切回训练模式
        
        # 打印结果
        #print(f"    Task {test_task_idx} - Accuracy: {accuracy:.6f}, NC1: {nc1:.6f}, NC2: {nc2:.6f}, NC3: {nc3:.6f}, NC4: {nc4:.6f}")
        print(f"    Task {test_task_idx} - Accuracy: {accuracy:.6f}, NC1: {nc1:.6f}, NC2: {nc2:.6f}, NC3: {nc3:.6f}, NC4: {nc4:.6f}, 等范数性: {equinormity:.6f}, 等角性: {isotropy:.6f}")
        # 写入CSV文件
        with open(forward_test_file, 'a', encoding='utf-8') as f:
            f.write(
                f"{current_task_idx},"          # 当前训练完成的任务ID
                f"{test_task_idx},"             # 被测试的历史任务ID
                f"{accuracy:.6f},"              # 准确率
                f"{nc1:.6f},{nc2:.6f},{nc3:.6f},{nc4:.6f},{isotropy:.6f},{equinormity:.6f},"  # NC指标
                f"{total_samples}\n"            # 样本数
            )
        
        # 释放内存
        release_tensor_memory(x_task, y_task)
    
    learner.net.train()
    print(f"=== Forward Test completed! Results saved to {forward_test_file} ===")
# ====================== 新增结束 ======================

def test_joint_model_per_task(joint_learner, tasks_permutations, x_original, y_original, save_dir, num_tasks, dev):
    joint_learner.net.eval()
    
    per_task_acc_file = os.path.join(save_dir, f'joint_model_per_task_accuracy_{num_tasks}_tasks.csv')
    
    with open(per_task_acc_file, 'w', encoding='utf-8') as f:
        f.write('task_idx,accuracy,num_samples,num_tasks,nc1,nc2,nc3,nc4,isotropy,equinormity\n')
    
    examples_per_task = EXAMPLES_PER_TASK
    print(f"\n[Joint Model Per-Task Test] Start testing {num_tasks} tasks (each with {EXAMPLES_PER_TASK} samples)...")
    for task_idx, (pixel_perm, data_perm) in enumerate(tasks_permutations):
        print(f"  Testing Task {task_idx}...")
        
        x_task, y_task = generate_task_samples(
            x_original, y_original, pixel_perm, data_perm, examples_per_task, dev
        )

        # ===== 分批前向（ResNet 一次性对 50000 张图前向会 OOM）=====
        task_dataset = TensorDataset(x_task, y_task)
        task_dataloader = DataLoader(task_dataset, batch_size=EVAL_BATCH_SIZE)
        with torch.no_grad():
            total_correct = 0
            for val_x, val_y in task_dataloader:
                val_output = joint_learner.net(val_x)
                preds = torch.argmax(val_output, dim=1)
                total_correct += (preds == val_y).sum().item()
            total_samples = y_task.size(0)
            accuracy = total_correct / total_samples

            nc1, nc2, nc3, nc3_max, nc3_min, nc4, isotropy, equinormity = NC(model=joint_learner.net, data_loader=task_dataloader, num_classes=10)
            
            torch.cuda.empty_cache()
            gc.collect()
            # del all_feats, all_labels, class_centers, classifier_weight  # 把 NC 里的大张量都删掉
            # torch.cuda.empty_cache()
            # gc.collect()
            #learner.net.train()  # 切回训练模式
        
        print(f"  Task {task_idx} Accuracy: {accuracy:.6f} (Correct: {total_correct}/{total_samples})")
        #print(f"  Task {task_idx} NC1: {nc1:.6f}, NC2: {nc2:.6f}, NC3: {nc3:.6f}, NC4: {nc4:.6f}")
        print(f"  Task {task_idx} NC1: {nc1:.6f}, NC2: {nc2:.6f}, NC3: {nc3:.6f}, NC4: {nc4:.6f}, 等范数性: {equinormity:.6f}, 等角性: {isotropy:.6f}")
        with open(per_task_acc_file, 'a', encoding='utf-8') as f:
            f.write(f"{task_idx},{accuracy:.6f},{total_samples},{num_tasks},{nc1:.6f},{nc2:.6f},{nc3:.6f},{nc4:.6f},{isotropy:.6f},{equinormity:.6f}\n")
            
        release_tensor_memory(x_task, y_task)
    
    print(f"[Joint Model Per-Task Test] Results saved to: {per_task_acc_file}")
    
    joint_learner.net.train()

def create_model(params, input_size, classes_per_task, num_hidden_layers, num_features, dev):
    # ===================== ResNet-18 分支 =====================
    # 复用 lop/incremental_cifar/learners.py 的 learner 栈：
    #   bp/l2 -> BPLearner (SGD, l2 通过 weight_decay 实现)
    #   cbp   -> CBPLearner (ResGnT, 支持 4D 卷积特征的 generate-and-test)
    #   ewc   -> EWCLearner (ResNetEWC, 已兼容 (x,y) 元组 dataloader)
    # 注意：input_size / num_features / num_hidden_layers 对 ResNet 无意义
    if params.get('net_type', 'mlp') == 'resnet':
        net = build_resnet18(num_classes=classes_per_task, norm_layer=torch.nn.BatchNorm2d, in_channels=INPUT_CHANNELS)
        net.apply(kaiming_init_resnet_module)
        net.layers_to_log = []   # 让 to_log 分支的遍历空转，避免 AttributeError

        if params.get('to_perturb', False):
            print("[Warning] ResNet learner 栈不支持 to_perturb（shrink-and-perturb），已忽略。")

        step_size = params['step_size']
        if isinstance(step_size, list):
            step_size = step_size[0]

        agent = params['agent']
        agent_mapped = {'bp': 'bp', 'l2': 'bp', 'cbp': 'cbp', 'ewc': 'ewc'}.get(agent, agent)
        if agent not in agent_mapped:
            raise ValueError(f"net_type='resnet' 不支持 agent='{agent}'，可选: bp/l2/cbp/ewc")

        optim = torch.optim.SGD(
            net.parameters(),
            lr=step_size,
            momentum=params.get('momentum', 0.9),
            weight_decay=params.get('weight_decay', 0),
        )
        learner = build_learner(
            agent=agent_mapped,
            net=net,
            optim=optim,
            loss_fn=torch.nn.CrossEntropyLoss(),
            device=torch.device(dev) if not isinstance(dev, torch.device) else dev,
            # CBP 参数
            replacement_rate=params.get('replacement_rate', 1e-5),
            maturity_threshold=params.get('maturity_threshold', params.get('mt', 1000)),
            util_type=params.get('utility_function', 'contribution'),
            # EWC 参数
            ewc_lambda=params.get('ewc_lambda', 5000.0),
            fisher_sample_size=params.get('ewc_fisher_sample_size', 2000),
        )
        learner.net = learner.net.to(dev)
        learner.net.train()
        return learner
    # ===================== ResNet 分支结束 =====================

    if params['agent'] == 'linear':
        net = MyLinear(
            input_size=input_size, num_outputs=classes_per_task
        )
        net.layers_to_log = []
    else:
        net = DeepFFNN(input_size=input_size, num_features=num_features, 
                      num_outputs=classes_per_task, num_hidden_layers=num_hidden_layers)
    
    if params['agent'] in ['bp', 'linear', "l2"]:
        learner = Backprop(
            net=net,
            step_size=params['step_size'],
            opt=params['opt'],
            loss='nll',
            weight_decay=params.get('weight_decay', 0),
            device=dev,
            to_perturb=params.get('to_perturb', False),
            perturb_scale=params.get('perturb_scale', 0.1),
        )
    elif params['agent'] in ['cbp']:
        learner = ContinualBackprop(
            net=net,
            step_size=params['step_size'],
            opt=params['opt'],
            loss='nll',
            maturity_threshold=params.get('mt', 100),
            decay_rate=params.get('decay_rate', 0.99),
            util_type=params.get('util_type', 'adaptable_contribution'),
            accumulate=True,
            device=dev,
        )
    elif params['agent'] in ['ewc']:  # 新增EWC分支
        learner = EWC(
            net=net,
            step_size=params['step_size'],
            opt=params['opt'],
            loss='nll',
            ewc_lambda=params.get('ewc_lambda', 5000),  # 从配置文件中读取λ，默认5000
            weight_decay=params.get('weight_decay', 0), 
            device=dev,
            to_perturb=params.get('to_perturb', False),      # 如果 Backprop 需要
            perturb_scale=params.get('perturb_scale', 0.1),
        )
    learner.net = learner.net.to(dev)
    learner.net.train()  # 强制训练模式，防止被NC改成eval导致设备漂移
    return learner


def set_lr_for_epoch(learner, params, epoch_in_task, num_epochs_per_task):
    """
    按 task 内 epoch 设置学习率（移植自 incremental_cifar_experiment.set_lr）。
    支持 'cosine'（余弦退火，每 task 重置）与 'step'（阶梯衰减）；
    'none'/未配置时不做任何事（保持 MLP 原行为）。
    兼容新旧两种 learner：新栈优化器在 learner.optim，旧栈在 learner.opt。
    """
    schedule = params.get('lr_schedule', 'none')
    if schedule in (None, '', 'none'):
        return
    opt_obj = getattr(learner, 'optim', None) or getattr(learner, 'opt', None)
    if opt_obj is None:
        return

    base_lr = params['step_size']
    if isinstance(base_lr, list):
        base_lr = base_lr[0]

    if schedule == 'cosine':
        lr_min_ratio = params.get('lr_min_ratio', 0.01)
        progress = epoch_in_task / max(num_epochs_per_task - 1, 1)
        lr_min = base_lr * lr_min_ratio
        current_lr = lr_min + 0.5 * (base_lr - lr_min) * (1.0 + math.cos(math.pi * progress))
    elif schedule == 'step':
        current_lr = base_lr
        for milestone in sorted(params.get('lr_decay_milestones', [])):
            if epoch_in_task >= milestone:
                current_lr *= params.get('lr_decay_gamma', 0.5)
    else:
        return

    for g in opt_obj.param_groups:
        g['lr'] = current_lr

# def train_single_task(learner, x_original, y_original, task_idx, params, save_dir, traj_save_dir, 
#                      num_hidden_layers, input_size, examples_per_task, change_after, 
#                      mini_batch_size, rank_measure_period, iter, accuracies, 
#                      weight_mag_sum, effective_ranks, approximate_ranks, 
#                      approximate_ranks_abs, ranks, dead_neurons, dev, tasks_permutations):
#     # traj_recorder = TrajectoryMap(
#     #     model=learner.net,
#     #     step=params.get('traj_step', 100),
#     #     save_dir=traj_save_dir
#     # )
    
#     new_iter_start = iter
    
#     # pixel_permutation = np.random.permutation(input_size)
#     # data_permutation = np.random.permutation(len(y_original)
#     # )
#     pixel_permutation, data_permutation = tasks_permutations[task_idx]  
    
#     x_task, y_task = generate_task_samples(
#         x_original, y_original, pixel_permutation, data_permutation, examples_per_task, dev
#     )
#     dataset = TensorDataset(x_task, y_task)
#     dataloader = DataLoader(dataset, batch_size=1000)
    
#     if params['agent'] != 'linear':
#         with torch.no_grad():
#             new_idx = int(iter / rank_measure_period)
#             m = learner.net.predict(x_task[:20000])[1]
#             task_start_approx_ranks = []
#             task_start_dead_neurons = []
#             for rep_layer_idx in range(num_hidden_layers):
#                 ranks[new_idx][rep_layer_idx], effective_ranks[new_idx][rep_layer_idx], \
#                 approx_rank_val, approximate_ranks_abs[new_idx][rep_layer_idx] = \
#                     compute_matrix_rank_summaries(m=m[rep_layer_idx], use_scipy=True)
#                 task_start_approx_ranks.append(round(float(approx_rank_val.item()), 6))
#                 neuron_activation_sums = m[rep_layer_idx].abs().sum(dim=0)
#                 dead = (neuron_activation_sums < dead_neuron_threshold).sum()
#                 task_start_dead_neurons.append(int(dead.item()))
#             #print(f'[Task {task_idx}] Initial approximate ranks: {task_start_approx_ranks}, Initial dead neurons: {task_start_dead_neurons}')
#             del m
    
#     intermediate_file = os.path.join(save_dir, '0_0.csv')
#     if not os.path.exists(intermediate_file):
#         intermediate_headers = [
#             'task_idx', 'iter', 'nc1', 'nc2', 'nc3', 'nc3_max', 'nc3_min', 'nc4', 'isotropy', 'equinormity', 'full_accuracy',
            
#         ]
#         #'traj_avg_cos', 'traj_min_cos', 'traj_max_cos', 'traj_length'
#         with open(intermediate_file, 'w', encoding='utf-8') as f:
#             f.write(','.join(intermediate_headers) + '\n')
    
#     #num_epochs = 10
#     num_epochs = 10
#     task_reached_threshold = False  
#     total_train_steps = change_after * num_epochs
#     nc1_interval = 50000  
    
#     for epoch in range(num_epochs):
#         if task_reached_threshold:
#             break  
#         print(f"\n[Task {task_idx}] Epoch {epoch+1}/{num_epochs}")
        
#         epoch_permutation = np.random.permutation(examples_per_task)
#         x_epoch = x_task[epoch_permutation]
#         y_epoch = y_task[epoch_permutation]
        
#         for start_idx in tqdm(range(0, change_after, mini_batch_size), desc=f"Task {task_idx} Epoch {epoch+1} Training"):
#             start_idx = start_idx % examples_per_task
#             batch_x = x_epoch[start_idx: start_idx + mini_batch_size]
#             batch_y = y_epoch[start_idx: start_idx + mini_batch_size]
            
#             loss, network_output = learner.learn(batch_x.to(dev), batch_y.to(dev))
            
#             if params.get('to_log', False) and params['agent'] != 'linear':
#                 for idx, layer_idx in enumerate(learner.net.layers_to_log):
#                     weight_mag_sum[iter][idx] = learner.net.layers[layer_idx].weight.data.abs().sum()
            
#             with torch.no_grad():
#                 accuracies[iter] = nll_accuracy(softmax(network_output, dim=1), batch_y).cpu()
            
#            # traj_recorder.record_trajectory(current_step=iter, is_train=True)
            
#             if (iter - new_iter_start + 1) % nc1_interval == 0:
#                 #nc1, nc2, nc3, nc4 = NC(model=learner.net, data_loader=dataloader, num_classes=10)
#                 nc1, nc2, nc3, nc3_max, nc3_min, nc4, isotropy, equinormity = NC(model=learner.net, data_loader=dataloader, num_classes=10)
#                 # 改显存
#                 torch.cuda.empty_cache()
#                 gc.collect()
#                 #learner.net.train()  # 切回训练模式
#                 # del all_feats, all_labels, class_centers, classifier_weight  # 把 NC 里的大张量都删掉
#                 # torch.cuda.empty_cache()
#                 # gc.collect()




                
#                 total_correct = 0
#                 total_samples = 0
#                 with torch.no_grad():
#                     for val_x, val_y in dataloader:
#                         val_output = learner.net(val_x)
#                         preds = torch.argmax(val_output, dim=1)
#                         total_correct += (preds == val_y).sum().item()
#                         total_samples += val_y.size(0)
#                 full_accuracy = total_correct / total_samples
#                 #0.96
#                 if full_accuracy >= 0.56:  
#                     task_reached_threshold = True
#                     print(f"[Task {task_idx}] Reached target accuracy {full_accuracy:.4f}, stopping early")
                
#                 try:
#                     from lop.utils.neural_collapse import _extract_features, _get_feature_means, _get_classifier_weights
#                     import pickle

#                     with torch.no_grad():
#                         all_feats = []
#                         all_labels = []
#                         for x, y in dataloader:
#                             feat = _extract_features(learner.net, x)
#                             all_feats.append(feat.cpu())
#                             all_labels.append(y.cpu())

#                     all_feats = torch.cat(all_feats)
#                     all_labels = torch.cat(all_labels)

#                     mu_G, mu_c_dict = _get_feature_means(
#                         model=learner.net,
#                         data_loader=dataloader,
#                         num_classes=10,
#                         use_cache=False
#                     )
#                     class_centers = torch.stack([mu_c_dict[c] for c in range(10)])
#                     classifier_weight = _get_classifier_weights(learner.net)
                    

#                     # ====================== 【新增：抽取每类前200个样本特征】 ======================
#                     sample_features_list = []  # 存储每类样本特征
#                     sample_labels_list = []    # 存储对应标签
#                     for c in range(10):
#                          # 取出当前类的所有特征 & 标签
#                         class_mask = (all_labels == c)
#                         class_feats = all_feats[class_mask]
#                         class_lbls = all_labels[class_mask]
        
#                         # 只取前200个
#                         class_feats = class_feats[:200]
#                         class_lbls = class_lbls[:200]
        
#                         sample_features_list.append(class_feats)
#                         sample_labels_list.append(class_lbls)
    
#                         # 拼接成 [2000, D] 的张量
#                     sample_features = torch.cat(sample_features_list).numpy()
#                     sample_labels = torch.cat(sample_labels_list).numpy()
#     # ============================================================================


#                     # 每 5 个任务保存一次
#                     if (task_reached_threshold or epoch == num_epochs - 1) and task_idx % 5 == 0:
#                         save_data_dir = os.path.join(save_dir, "nc3_saved_data")
#                         os.makedirs(save_data_dir, exist_ok=True)
#                         save_dict = {
#                             "task_idx": task_idx,
#                             "class_centers": class_centers.detach().cpu().numpy(),
#                             "classifier_weights": classifier_weight.detach().cpu().numpy(),
#                             "sample_features": sample_features,    # 【新增】每类前200样本隐藏层特征 [2000, D]
#                             "sample_labels": sample_labels         # 【新增】对应标签 [2000]
#                         }
#                         save_path = os.path.join(save_data_dir, f"nc3_data_task_{task_idx}.pkl")
#                         with open(save_path, "wb") as f:
#                             pickle.dump(save_dict, f)
#                         print(f"✅ NC3 数据已保存：{save_path}")

#                     if task_reached_threshold or epoch == num_epochs - 1:    
#                         visualize_nc3_tsne(class_centers, classifier_weight, task_idx, os.path.join(save_dir, "nc3_tsne"))

#                 except Exception as e:
#                     print(f"[NC3 错误] {e}")
#                 # ========================================================================    
# # ====================== 加完 ======================    







#                 #current_traj_metrics = traj_recorder.compute_trajectory_metrics()
                
#                 #print(f"\n[Task {task_idx} Epoch {epoch+1} Iter {iter}] NC1: {nc1:.6f}, NC2: {nc2:.6f}, NC3: {nc3:.6f}, NC4: {nc4:.6f}")
#                 print(f"\n[Task {task_idx} Epoch {epoch+1} Iter {iter}] NC1: {nc1:.6f}, NC2: {nc2:.6f}, NC3: {nc3:.6f}, NC4: {nc4:.6f}, 等范数性: {equinormity:.6f}, 等角性: {isotropy:.6f}")
#                 print(f"[Task {task_idx} Epoch {epoch+1} Iter {iter}] Full Accuracy: {full_accuracy:.4f}")
#                 #print(f"[Task {task_idx} Epoch {epoch+1} Iter {iter}] Traj Metrics - Avg Cos: {current_traj_metrics['traj_avg_cos']:.6f}, Min Cos: {current_traj_metrics['traj_min_cos']:.6f}, Max Cos: {current_traj_metrics['traj_max_cos']:.6f}, Length: {current_traj_metrics['traj_length']}")
                
#                 with open(intermediate_file, 'a', encoding='utf-8') as f:
#                     f.write(
#                         f"{task_idx},{iter},{nc1:.6f},{nc2:.6f},{nc3:.6f},{nc3_max:.6f},{nc3_min:.6f},{nc4:.6f},{isotropy:.6f},{equinormity:.6f},{full_accuracy:.6f}\n"
#                        # f"{current_traj_metrics['traj_avg_cos']:.6f},{current_traj_metrics['traj_min_cos']:.6f},"
#                         #f"{current_traj_metrics['traj_max_cos']:.6f},{current_traj_metrics['traj_length']:.0f}\n"
#                     )
            
#             del batch_x, batch_y
#             iter += 1  
            
#             if iter >= len(accuracies) or task_reached_threshold:
#                 break
    
#     # task_traj_metrics = traj_recorder.compute_trajectory_metrics()
#     # task_total_iters = iter - new_iter_start
#     # traj_recorder.plot_trajectory_map(task_idx=task_idx, total_iterations=task_total_iters)
    
#     print("[DEBUG] After NC computation: model.training =", learner.net.training)
#     nc1, nc2, nc3, nc3_max, nc3_min, nc4, isotropy, equinormity = NC(model=learner.net, data_loader=dataloader, num_classes=10)
#     # 改显存
#     torch.cuda.empty_cache()
#     gc.collect()
#     # del all_feats, all_labels, class_centers, classifier_weight  # 把 NC 里的大张量都删掉
#     # torch.cuda.empty_cache()
#     # gc.collect()

#    # learner.net.train()  # 切回训练模式
#     print("[DEBUG] After net.train(): model.training =", learner.net.training)
#     #print(f"[Task {task_idx} End] NC1: {nc1:.6f}, NC2: {nc2:.6f}, NC3: {nc3:.6f}, NC4: {nc4:.6f}")
#     print(f"[Task {task_idx} End] NC1: {nc1:.6f}, NC2: {nc2:.6f}, NC3: {nc3:.6f}, NC4: {nc4:.6f}, 等范数性: {equinormity:.6f}, 等角性: {isotropy:.6f}")

#     if params['agent'] != 'linear':
#         if len(task_start_approx_ranks) != num_hidden_layers or len(task_start_dead_neurons) != num_hidden_layers:
#             task_approx_ranks = [0.0] * num_hidden_layers
#             task_dead_neurons = calculate_dead_neurons(learner, x_task, num_hidden_layers, dev)
#         else:
#             task_approx_ranks = task_start_approx_ranks
#             task_dead_neurons = calculate_dead_neurons(learner, x_task, num_hidden_layers, dev)
#     else:
#         task_approx_ranks = [0.0] * num_hidden_layers
#         task_dead_neurons = [0] * num_hidden_layers
    
#     recent_acc = float(accuracies[new_iter_start:iter - 1].mean().item())
    
#     task_end_file = os.path.join(save_dir, '0.csv')
#     if not os.path.exists(task_end_file):
#         approx_rank_headers = [f"approx_rank_{i+1}" for i in range(num_hidden_layers)]
#         dead_neurons_headers = [f"dead_neurons_{i+1}" for i in range(num_hidden_layers)]
#         all_headers = [
#             'task_idx', 'nc1', 'nc2', 'nc3',  'nc3_max', 'nc3_min', 'nc4', 'isotropy', 'equinormity', 'recent_accuracy', 'reached_threshold'
#         ] + approx_rank_headers + dead_neurons_headers 
#         #+ [
#            # 'traj_avg_cos', 'traj_min_cos', 'traj_max_cos', 'traj_length'
#         #]
#         with open(task_end_file, 'w', encoding='utf-8') as f:
#             f.write(','.join(all_headers) + '\n')
    
#     with open(task_end_file, 'a', encoding='utf-8') as f:
#         data_row = [
#             str(task_idx),
#             f"{nc1:.6f}", f"{nc2:.6f}", f"{nc3:.6f}",f"{nc3_max:.6f}", f"{nc3_min:.6f}", f"{nc4:.6f}", f"{isotropy:.6f}", f"{equinormity:.6f}",
#             f"{recent_acc:.6f}", str(task_reached_threshold)
#         ] + [f"{ar}" for ar in task_approx_ranks] + [f"{dn}" for dn in task_dead_neurons] + [
#            # f"{task_traj_metrics['traj_avg_cos']:.6f}",
#             #f"{task_traj_metrics['traj_min_cos']:.6f}",
#             #f"{task_traj_metrics['traj_max_cos']:.6f}",
#             #f"{task_traj_metrics['traj_length']:.0f}"
#         ]
#         f.write(','.join(data_row) + '\n')
    
#     # ====================== EWC: 任务结束后更新 Fisher 矩阵 ======================
#     if params['agent'] == 'ewc':
#         # 使用当前任务的数据加载器 dataloader 来估计 Fisher 信息矩阵
#         learner.after_task(dataloader)
#         print(f"[Task {task_idx}] EWC Fisher matrix and previous params updated.")
#     # ============================================================================


#     release_tensor_memory(x_task, y_task, x_epoch, y_epoch)
#     return iter, learner, pixel_permutation, data_permutation, task_reached_threshold


def train_single_task(learner, x_original, y_original, task_idx, params, save_dir, traj_save_dir, 
                     num_hidden_layers, input_size, examples_per_task, change_after, 
                     mini_batch_size, rank_measure_period, iter, accuracies, 
                     weight_mag_sum, effective_ranks, approximate_ranks, 
                     approximate_ranks_abs, ranks, dead_neurons, dev, tasks_permutations,
                     plasticity_file, skip_threshold):
    # traj_recorder = TrajectoryMap(
    #     model=learner.net,
    #     step=params.get('traj_step', 100),
    #     save_dir=traj_save_dir
    # )
    
    new_iter_start = iter
    task_reached_threshold = False
    pixel_permutation, data_permutation = tasks_permutations[task_idx]  
    
    x_task, y_task = generate_task_samples(
        x_original, y_original, pixel_permutation, data_permutation, examples_per_task, dev
    )
    dataset = TensorDataset(x_task, y_task)
    dataloader = DataLoader(dataset, batch_size=EVAL_BATCH_SIZE)

    # ========== 预评估及可塑性指标（开始） ==========
    criterion = F.cross_entropy   # 与 Backprop 中的 F.cross_entropy 一致
    
    loss_pre = compute_avg_loss(learner.net, dataloader, dev, loss_func=criterion)
    print(f"[Task {task_idx}] Pre-assessment loss: {loss_pre:.6f}")
    
    if loss_pre <= skip_threshold:
        print(f"[Task {task_idx}] Skipped (loss_pre <= {skip_threshold})")
        # 跳过任务，不改变模型，不增加 iter，返回 skip=True
        return iter, learner, pixel_permutation, data_permutation, task_reached_threshold, True
    
    # 保存当前参数（θ_{n-1}^*）
    params_before = copy.deepcopy(learner.net.state_dict())
    # ========== 预评估结束 ==========
    
    task_start_approx_ranks = []
    task_start_dead_neurons = []
    # ResNet 无 predict() 接口且秩/死亡神经元统计对 4D 卷积特征不成立，跳过（CSV 以 0 占位）
    if params['agent'] != 'linear' and hasattr(learner.net, 'predict'):
        with torch.no_grad():
            new_idx = int(iter / rank_measure_period)
            m = learner.net.predict(x_task[:20000])[1]
            for rep_layer_idx in range(num_hidden_layers):
                ranks[new_idx][rep_layer_idx], effective_ranks[new_idx][rep_layer_idx], \
                approx_rank_val, approximate_ranks_abs[new_idx][rep_layer_idx] = \
                    compute_matrix_rank_summaries(m=m[rep_layer_idx], use_scipy=True)
                task_start_approx_ranks.append(round(float(approx_rank_val.item()), 6))
                neuron_activation_sums = m[rep_layer_idx].abs().sum(dim=0)
                dead = (neuron_activation_sums < dead_neuron_threshold).sum()
                task_start_dead_neurons.append(int(dead.item()))
            del m
    
    intermediate_file = os.path.join(save_dir, '0_0.csv')
    if not os.path.exists(intermediate_file):
        intermediate_headers = [
            'task_idx', 'iter', 'nc1', 'nc2', 'nc3', 'nc3_max', 'nc3_min', 'nc4', 'isotropy', 'equinormity', 'full_accuracy',
        ]
        with open(intermediate_file, 'w', encoding='utf-8') as f:
            f.write(','.join(intermediate_headers) + '\n')
    
    num_epochs = NUM_EPOCHS
    #task_reached_threshold = False
    total_train_steps = change_after * num_epochs
    # NC 记录间隔按 mini-batch 迭代数计：batch 变大后每 epoch 迭代数变少，
    # 需在配置里相应调小（如 bs=128 时每 epoch 约 391 次迭代）
    nc1_interval = NC1_INTERVAL

    # ========== 第一步：单独执行第一个 epoch，用于计算可塑性指标 ==========
    epoch = 0
    print(f"\n[Task {task_idx}] Epoch {epoch+1}/{num_epochs} (Plasticity measurement epoch)")
    set_lr_for_epoch(learner, params, epoch, num_epochs)
    
    epoch_permutation = np.random.permutation(examples_per_task)
    x_epoch = x_task[epoch_permutation]
    y_epoch = y_task[epoch_permutation]
    
    # 训练第一个 epoch
    for start_idx in tqdm(range(0, change_after, mini_batch_size), desc=f"Task {task_idx} Epoch {epoch+1} Training"):
        start_idx = start_idx % examples_per_task
        batch_x = x_epoch[start_idx: start_idx + mini_batch_size]
        batch_y = y_epoch[start_idx: start_idx + mini_batch_size]
        
        loss, network_output = learner.learn(batch_x.to(dev), batch_y.to(dev))
        
        if params.get('to_log', False) and params['agent'] != 'linear':
            for idx, layer_idx in enumerate(learner.net.layers_to_log):
                weight_mag_sum[iter][idx] = learner.net.layers[layer_idx].weight.data.abs().sum()
        
        with torch.no_grad():
            accuracies[iter] = nll_accuracy(softmax(network_output, dim=1), batch_y).cpu()
        
        if (iter - new_iter_start + 1) % nc1_interval == 0:
            nc1, nc2, nc3, nc3_max, nc3_min, nc4, isotropy, equinormity = NC(model=learner.net, data_loader=dataloader, num_classes=10)
            torch.cuda.empty_cache()
            gc.collect()
            
            total_correct = 0
            total_samples = 0
            with torch.no_grad():
                for val_x, val_y in dataloader:
                    val_output = learner.net(val_x)
                    preds = torch.argmax(val_output, dim=1)
                    total_correct += (preds == val_y).sum().item()
                    total_samples += val_y.size(0)
            full_accuracy = total_correct / total_samples
            if full_accuracy >= TASK_ACC_THRESHOLD:  
                task_reached_threshold = True
                print(f"[Task {task_idx}] Reached target accuracy {full_accuracy:.4f}, stopping early")
            
            try:
                from lop.utils.neural_collapse import _extract_features, _get_feature_means, _get_classifier_weights
                import pickle

                with torch.no_grad():
                    all_feats = []
                    all_labels = []
                    for x, y in dataloader:
                        feat = _extract_features(learner.net, x)
                        all_feats.append(feat.cpu())
                        all_labels.append(y.cpu())

                all_feats = torch.cat(all_feats)
                all_labels = torch.cat(all_labels)

                mu_G, mu_c_dict = _get_feature_means(
                    model=learner.net,
                    data_loader=dataloader,
                    num_classes=10,
                    use_cache=False
                )
                class_centers = torch.stack([mu_c_dict[c] for c in range(10)])
                classifier_weight = _get_classifier_weights(learner.net)
                
                sample_features_list = []
                sample_labels_list = []
                for c in range(10):
                    class_mask = (all_labels == c)
                    class_feats = all_feats[class_mask]
                    class_lbls = all_labels[class_mask]
                    class_feats = class_feats[:200]
                    class_lbls = class_lbls[:200]
                    sample_features_list.append(class_feats)
                    sample_labels_list.append(class_lbls)
                sample_features = torch.cat(sample_features_list).numpy()
                sample_labels = torch.cat(sample_labels_list).numpy()

                if (task_reached_threshold or epoch == num_epochs - 1) and task_idx % 1 == 0:
                    save_data_dir = os.path.join(save_dir, "nc3_saved_data")
                    os.makedirs(save_data_dir, exist_ok=True)
                    save_dict = {
                        "task_idx": task_idx,
                        "class_centers": class_centers.detach().cpu().numpy(),
                        "classifier_weights": classifier_weight.detach().cpu().numpy(),
                        "sample_features": sample_features,
                        "sample_labels": sample_labels
                    }
                    save_path = os.path.join(save_data_dir, f"nc3_data_task_{task_idx}.pkl")
                    with open(save_path, "wb") as f:
                        pickle.dump(save_dict, f)
                    print(f"✅ NC3 数据已保存：{save_path}")

                if task_reached_threshold or epoch == num_epochs - 1:    
                    visualize_nc3_tsne(class_centers, classifier_weight, task_idx, os.path.join(save_dir, "nc3_tsne"))

            except Exception as e:
                print(f"[NC3 错误] {e}")
            
            print(f"\n[Task {task_idx} Epoch {epoch+1} Iter {iter}] NC1: {nc1:.6f}, NC2: {nc2:.6f}, NC3: {nc3:.6f}, NC4: {nc4:.6f}, 等范数性: {equinormity:.6f}, 等角性: {isotropy:.6f}")
            print(f"[Task {task_idx} Epoch {epoch+1} Iter {iter}] Full Accuracy: {full_accuracy:.4f}")
            
            with open(intermediate_file, 'a', encoding='utf-8') as f:
                f.write(
                    f"{task_idx},{iter},{nc1:.6f},{nc2:.6f},{nc3:.6f},{nc3_max:.6f},{nc3_min:.6f},{nc4:.6f},{isotropy:.6f},{equinormity:.6f},{full_accuracy:.6f}\n"
                )
        
        del batch_x, batch_y
        iter += 1  
        
        if iter >= len(accuracies) :
            break
    
    # 第一个 epoch 结束后，计算可塑性指标
    loss_post = compute_avg_loss(learner.net, dataloader, dev, loss_func=criterion)
    params_after = learner.net.state_dict()
    delta_norm_total, layer_norms = compute_param_diff_norm(params_before, params_after)
    P_n = (loss_pre - loss_post) / (delta_norm_total + 1e-8)
    
    with open(plasticity_file, 'a', encoding='utf-8') as f:
        f.write(f"{task_idx},{loss_pre:.6f},{loss_post:.6f},{delta_norm_total:.6f},{P_n:.6f}\n")
    
    print(f"[Task {task_idx}] Plasticity P_n = {P_n:.6f} (ΔL={loss_pre-loss_post:.4f}, Δθ={delta_norm_total:.4f})")
    
    # ========== 第二步：如果第一个 epoch 未达到阈值，继续剩余 epoch ==========
    if not task_reached_threshold:
        for epoch in range(1, num_epochs):
            if task_reached_threshold:
                break
            print(f"\n[Task {task_idx}] Epoch {epoch+1}/{num_epochs}")
            set_lr_for_epoch(learner, params, epoch, num_epochs)

            epoch_permutation = np.random.permutation(examples_per_task)
            x_epoch = x_task[epoch_permutation]
            y_epoch = y_task[epoch_permutation]
            
            for start_idx in tqdm(range(0, change_after, mini_batch_size), desc=f"Task {task_idx} Epoch {epoch+1} Training"):
                start_idx = start_idx % examples_per_task
                batch_x = x_epoch[start_idx: start_idx + mini_batch_size]
                batch_y = y_epoch[start_idx: start_idx + mini_batch_size]
                
                loss, network_output = learner.learn(batch_x.to(dev), batch_y.to(dev))
                
                if params.get('to_log', False) and params['agent'] != 'linear':
                    for idx, layer_idx in enumerate(learner.net.layers_to_log):
                        weight_mag_sum[iter][idx] = learner.net.layers[layer_idx].weight.data.abs().sum()
                
                with torch.no_grad():
                    accuracies[iter] = nll_accuracy(softmax(network_output, dim=1), batch_y).cpu()
                
                if (iter - new_iter_start + 1) % nc1_interval == 0:
                    nc1, nc2, nc3, nc3_max, nc3_min, nc4, isotropy, equinormity = NC(model=learner.net, data_loader=dataloader, num_classes=10)
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    total_correct = 0
                    total_samples = 0
                    with torch.no_grad():
                        for val_x, val_y in dataloader:
                            val_output = learner.net(val_x)
                            preds = torch.argmax(val_output, dim=1)
                            total_correct += (preds == val_y).sum().item()
                            total_samples += val_y.size(0)
                    full_accuracy = total_correct / total_samples
                    if full_accuracy >= TASK_ACC_THRESHOLD:  
                        task_reached_threshold = True
                        print(f"[Task {task_idx}] Reached target accuracy {full_accuracy:.4f}, stopping early")
                    
                    try:
                        from lop.utils.neural_collapse import _extract_features, _get_feature_means, _get_classifier_weights
                        import pickle

                        with torch.no_grad():
                            all_feats = []
                            all_labels = []
                            for x, y in dataloader:
                                feat = _extract_features(learner.net, x)
                                all_feats.append(feat.cpu())
                                all_labels.append(y.cpu())

                        all_feats = torch.cat(all_feats)
                        all_labels = torch.cat(all_labels)

                        mu_G, mu_c_dict = _get_feature_means(
                            model=learner.net,
                            data_loader=dataloader,
                            num_classes=10,
                            use_cache=False
                        )
                        class_centers = torch.stack([mu_c_dict[c] for c in range(10)])
                        classifier_weight = _get_classifier_weights(learner.net)
                        
                        sample_features_list = []
                        sample_labels_list = []
                        for c in range(10):
                            class_mask = (all_labels == c)
                            class_feats = all_feats[class_mask]
                            class_lbls = all_labels[class_mask]
                            class_feats = class_feats[:200]
                            class_lbls = class_lbls[:200]
                            sample_features_list.append(class_feats)
                            sample_labels_list.append(class_lbls)
                        sample_features = torch.cat(sample_features_list).numpy()
                        sample_labels = torch.cat(sample_labels_list).numpy()

                        if (task_reached_threshold or epoch == num_epochs - 1) and task_idx % 1 == 0:
                            save_data_dir = os.path.join(save_dir, "nc3_saved_data")
                            os.makedirs(save_data_dir, exist_ok=True)
                            save_dict = {
                                "task_idx": task_idx,
                                "class_centers": class_centers.detach().cpu().numpy(),
                                "classifier_weights": classifier_weight.detach().cpu().numpy(),
                                "sample_features": sample_features,
                                "sample_labels": sample_labels
                            }
                            save_path = os.path.join(save_data_dir, f"nc3_data_task_{task_idx}.pkl")
                            with open(save_path, "wb") as f:
                                pickle.dump(save_dict, f)
                            print(f"✅ NC3 数据已保存：{save_path}")

                        if task_reached_threshold or epoch == num_epochs - 1:    
                            visualize_nc3_tsne(class_centers, classifier_weight, task_idx, os.path.join(save_dir, "nc3_tsne"))

                    except Exception as e:
                        print(f"[NC3 错误] {e}")
                    
                    print(f"\n[Task {task_idx} Epoch {epoch+1} Iter {iter}] NC1: {nc1:.6f}, NC2: {nc2:.6f}, NC3: {nc3:.6f}, NC4: {nc4:.6f}, 等范数性: {equinormity:.6f}, 等角性: {isotropy:.6f}")
                    print(f"[Task {task_idx} Epoch {epoch+1} Iter {iter}] Full Accuracy: {full_accuracy:.4f}")
                    
                    with open(intermediate_file, 'a', encoding='utf-8') as f:
                        f.write(
                            f"{task_idx},{iter},{nc1:.6f},{nc2:.6f},{nc3:.6f},{nc3_max:.6f},{nc3_min:.6f},{nc4:.6f},{isotropy:.6f},{equinormity:.6f},{full_accuracy:.6f}\n"
                        )
                
                del batch_x, batch_y
                iter += 1  
                
                if iter >= len(accuracies) or task_reached_threshold:
                    break
    
    # 任务结束后的 NC 计算和日志记录（原代码保持不变）
    print("[DEBUG] After NC computation: model.training =", learner.net.training)
    nc1, nc2, nc3, nc3_max, nc3_min, nc4, isotropy, equinormity = NC(model=learner.net, data_loader=dataloader, num_classes=10)
    torch.cuda.empty_cache()
    gc.collect()
    print("[DEBUG] After net.train(): model.training =", learner.net.training)
    print(f"[Task {task_idx} End] NC1: {nc1:.6f}, NC2: {nc2:.6f}, NC3: {nc3:.6f}, NC4: {nc4:.6f}, 等范数性: {equinormity:.6f}, 等角性: {isotropy:.6f}")

    if params['agent'] != 'linear':
        if len(task_start_approx_ranks) != num_hidden_layers or len(task_start_dead_neurons) != num_hidden_layers:
            task_approx_ranks = [0.0] * num_hidden_layers
            task_dead_neurons = calculate_dead_neurons(learner, x_task, num_hidden_layers, dev)
        else:
            task_approx_ranks = task_start_approx_ranks
            task_dead_neurons = calculate_dead_neurons(learner, x_task, num_hidden_layers, dev)
    else:
        task_approx_ranks = [0.0] * num_hidden_layers
        task_dead_neurons = [0] * num_hidden_layers
    
    recent_acc = float(accuracies[new_iter_start:iter - 1].mean().item())
    
    task_end_file = os.path.join(save_dir, '0.csv')
    if not os.path.exists(task_end_file):
        approx_rank_headers = [f"approx_rank_{i+1}" for i in range(num_hidden_layers)]
        dead_neurons_headers = [f"dead_neurons_{i+1}" for i in range(num_hidden_layers)]
        all_headers = [
            'task_idx', 'nc1', 'nc2', 'nc3',  'nc3_max', 'nc3_min', 'nc4', 'isotropy', 'equinormity', 'recent_accuracy', 'reached_threshold'
        ] + approx_rank_headers + dead_neurons_headers 
        with open(task_end_file, 'w', encoding='utf-8') as f:
            f.write(','.join(all_headers) + '\n')
    
    with open(task_end_file, 'a', encoding='utf-8') as f:
        data_row = [
            str(task_idx),
            f"{nc1:.6f}", f"{nc2:.6f}", f"{nc3:.6f}",f"{nc3_max:.6f}", f"{nc3_min:.6f}", f"{nc4:.6f}", f"{isotropy:.6f}", f"{equinormity:.6f}",
            f"{recent_acc:.6f}", str(task_reached_threshold)
        ] + [f"{ar}" for ar in task_approx_ranks] + [f"{dn}" for dn in task_dead_neurons]
        f.write(','.join(data_row) + '\n')
    
    if params['agent'] == 'ewc':
        learner.after_task(dataloader)
        print(f"[Task {task_idx}] EWC Fisher matrix and previous params updated.")
    
        # ====================== 每个任务结束时，为 task_idx % 5 == 0 的任务生成最终 t-SNE 图 ======================
    if task_idx % 1 == 0:
        try:
            from lop.utils.neural_collapse import _extract_features, _get_feature_means, _get_classifier_weights
            # 重新计算一次 NC（为了获得最新的类中心和分类器权重）
            nc1, nc2, nc3, nc3_max, nc3_min, nc4, isotropy, equinormity = NC(
                model=learner.net, data_loader=dataloader, num_classes=10
            )
            torch.cuda.empty_cache()
            gc.collect()

            # 提取类中心和分类器权重
            mu_G, mu_c_dict = _get_feature_means(
                model=learner.net,
                data_loader=dataloader,
                num_classes=10,
                use_cache=False
            )
            class_centers = torch.stack([mu_c_dict[c] for c in range(10)])
            classifier_weight = _get_classifier_weights(learner.net)

            # 生成并保存 t-SNE 图
            visualize_nc3_tsne(
                class_centers, classifier_weight, task_idx,
                os.path.join(save_dir, "nc3_tsne")
            )
        except Exception as e:
            print(f"[NC3 生成错误] task {task_idx}: {e}")
    # =======================================================================================================



    release_tensor_memory(x_task, y_task, x_epoch, y_epoch)
    return iter, learner, pixel_permutation, data_permutation, task_reached_threshold, False





def train_joint_model(params, tasks_permutations, x_original, y_original, save_dir, dev):
    n_tasks = len(tasks_permutations)
    samples_per_task = EXAMPLES_PER_TASK
    total_samples_per_epoch = n_tasks * samples_per_task
    mini_batch_size = params.get('mini_batch_size', 1)
    if NET_TYPE == 'resnet' and mini_batch_size < 16:
        mini_batch_size = 128   # BatchNorm 限制，与 online_expr 中的守卫保持一致
    #num_epochs = 10  0
    num_epochs = NUM_EPOCHS
    record_interval = total_samples_per_epoch // 1  # joint training 数据量大(180万)，每 epoch 评估 1 次 NC
    record_points = [i * record_interval for i in range(1, 2)]  
    
    joint_intermediate_file = os.path.join(save_dir, 'joint_intermediate_results.csv')
    if not os.path.exists(joint_intermediate_file):
        intermediate_headers = [
            'epoch', 'record_idx_in_epoch', 'total_trained_samples', 
            'accuracy', 'nc1', 'nc2', 'nc3', 'nc4', 'isotropy', 'equinormity', 'num_tasks'
        ]
        with open(joint_intermediate_file, 'w', encoding='utf-8') as f:
            f.write(','.join(intermediate_headers) + '\n')

    joint_x_list = []
    joint_y_list = []
    examples_per_task = EXAMPLES_PER_TASK
    for pixel_perm, data_perm in tasks_permutations:
        x_task, y_task = generate_task_samples(
            x_original, y_original, pixel_perm, data_perm, examples_per_task, dev
        )
        joint_x_list.append(x_task.to(dev))
        joint_y_list.append(y_task.to(dev))
    joint_x = torch.cat(joint_x_list).to(dev)  # 最终拼接后再确认
    joint_y = torch.cat(joint_y_list).to(dev)
    
    release_tensor_memory(*joint_x_list, *joint_y_list)
    
    # NC 评估用子集（180万样本全量评估太慢 ~20min，取 10万样本子集 ~1min）
    nc_subset_size = min(100000, len(joint_x))
    nc_indices = torch.randperm(len(joint_x))[:nc_subset_size]
    nc_dataset = TensorDataset(joint_x[nc_indices], joint_y[nc_indices])
    nc_dataloader = DataLoader(nc_dataset, batch_size=EVAL_BATCH_SIZE)
    print(f"[Joint Training] NC eval subset: {nc_subset_size} / {len(joint_x)} samples")  
    
    input_size = joint_x.shape[1]
    classes_per_task = 10
    num_hidden_layers = params.get('num_hidden_layers', 1)
    num_features = params.get('num_features', 200)#wang
    learner = create_model(params, input_size, classes_per_task, num_hidden_layers, num_features, dev)
    
    joint_reached_threshold = False
    epoch_iter = 0  
    total_trained_samples_global = 0  
    
    for epoch in range(num_epochs):
        if joint_reached_threshold:
            break
        
        epoch_perm = np.random.permutation(total_samples_per_epoch)
        x_epoch = joint_x[epoch_perm]
        y_epoch = joint_y[epoch_perm]
        
        print(f"\nJoint Training Epoch {epoch+1}/{num_epochs}")
        set_lr_for_epoch(learner, params, epoch, num_epochs)
        current_total_samples = 0
        recorded_points = set()
        
        pbar = tqdm(
            range(0, total_samples_per_epoch, mini_batch_size),
            desc=f"Joint Training Epoch {epoch+1} Training"
        )
        
        for start_idx in pbar:
            batch_x = x_epoch[start_idx: start_idx + mini_batch_size]
            batch_y = y_epoch[start_idx: start_idx + mini_batch_size]
            if len(batch_x) == 0:
                continue
            
            learner.learn(batch_x.to(dev), batch_y.to(dev))
            epoch_iter += 1
            current_total_samples += len(batch_x)  
            total_trained_samples_global += len(batch_x)  
            
            for point in record_points:
                if point not in recorded_points and current_total_samples >= point:
                    nc1, nc2, nc3, nc3_max, nc3_min, nc4, isotropy, equinormity = NC(model=learner.net, data_loader=nc_dataloader, num_classes=10)
                    # 改显存
                    torch.cuda.empty_cache()
                    gc.collect()
                    #learner.net.train()  # 切回训练模式
                    # del all_feats, all_labels, class_centers, classifier_weight  # 把 NC 里的大张量都删掉
                    # torch.cuda.empty_cache()
                    # gc.collect()
                    
                    total_correct = 0
                    total_samples_eval = 0
                    with torch.no_grad():
                        for val_x, val_y in nc_dataloader:
                            val_output = learner.net(val_x)
                            preds = torch.argmax(val_output, dim=1)
                            total_correct += (preds == val_y).sum().item()
                            total_samples_eval += val_y.size(0)
                    full_accuracy = total_correct / total_samples_eval
                    
                    #print(f"\n[Joint Training Epoch {epoch+1} Iter {epoch_iter}] NC1: {nc1:.6f}, NC2: {nc2:.6f}, NC3: {nc3:.6f}, NC4: {nc4:.6f}")
                    print(f"\n[Joint Training Epoch {epoch+1} Iter {epoch_iter}] NC1: {nc1:.6f}, NC2: {nc2:.6f}, NC3: {nc3:.6f}, NC4: {nc4:.6f}, 等范数性: {equinormity:.6f}, 等角性: {isotropy:.6f}")
                    print(f"[Joint Training Epoch {epoch+1} Iter {epoch_iter}] Full Accuracy: {full_accuracy:.4f}")
                    
                    record_idx_in_epoch = len(recorded_points) + 1
                    
                    with open(joint_intermediate_file, 'a', encoding='utf-8') as f:
                        f.write(
                            f"{epoch+1},{record_idx_in_epoch},{total_trained_samples_global},"
                            f"{full_accuracy:.6f},{nc1:.6f},{nc2:.6f},{nc3:.6f},{nc4:.6f},{isotropy:.6f},{equinormity:.6f},{n_tasks}\n"
                        )
                    #0.96
                    if full_accuracy >= TASK_ACC_THRESHOLD:
                        joint_reached_threshold = True
                        print(f"Joint Training Reached target accuracy {full_accuracy:.4f}, stopping early")
                        pbar.close()  
                        break
                    
                    recorded_points.add(point)  
            if joint_reached_threshold:
                break
            
            del batch_x, batch_y
        
        if joint_reached_threshold:
            break
    
    nc1, nc2, nc3, nc3_max, nc3_min, nc4, isotropy, equinormity = NC(model=learner.net, data_loader=nc_dataloader, num_classes=10)
    # 改显存
    torch.cuda.empty_cache()
    gc.collect()
    # del all_feats, all_labels, class_centers, classifier_weight  # 把 NC 里的大张量都删掉
    # torch.cuda.empty_cache()
    # gc.collect()
    #learner.net.train()  # 切回训练模式
    total_correct = 0
    total_samples_eval = 0
    with torch.no_grad():
        for val_x, val_y in nc_dataloader:
            val_output = learner.net(val_x)
            preds = torch.argmax(val_output, dim=1)
            total_correct += (preds == val_y).sum().item()
            total_samples_eval += val_y.size(0)
    final_accuracy = total_correct / total_samples_eval
    #print(f"NC1: {nc1:.6f}, NC2: {nc2:.6f}, NC3: {nc3:.6f}, NC4: {nc4:.6f}")
    print(f"[Joint Training End] NC1: {nc1:.6f}, NC2: {nc2:.6f}, NC3: {nc3:.6f}, NC4: {nc4:.6f}, 等范数性: {equinormity:.6f}, 等角性: {isotropy:.6f}")

    release_tensor_memory(joint_x, joint_y, x_epoch, y_epoch)
    
    return learner, final_accuracy, nc1, nc2, nc3, nc4, isotropy, equinormity, joint_reached_threshold

def train_independent_tasks(params, tasks_permutations, x_original, y_original, save_dir, traj_save_dir, 
                           num_hidden_layers, input_size, examples_per_task, change_after, 
                           mini_batch_size, dev):
    independent_model_dir = os.path.join(save_dir, 'independent_models')
    os.makedirs(independent_model_dir, exist_ok=True)
    
    independent_intermediate_file = os.path.join(save_dir, 'independent_intermediate_results.csv')
    if not os.path.exists(independent_intermediate_file):
        intermediate_headers = [
            'task_idx', 'iter', 'nc1', 'nc2', 'nc3', 'nc4', 'isotropy', 'equinormity', 'full_accuracy',
           
        ]
         #'traj_avg_cos', 'traj_min_cos', 'traj_max_cos', 'traj_length'
        with open(independent_intermediate_file, 'w', encoding='utf-8') as f:
            f.write(','.join(intermediate_headers) + '\n')
    
    independent_final_file = os.path.join(save_dir, 'independent_final_results.csv')
    if not os.path.exists(independent_final_file):
        approx_rank_headers = [f"approx_rank_{i+1}" for i in range(num_hidden_layers)]
        dead_neurons_headers = [f"dead_neurons_{i+1}" for i in range(num_hidden_layers)]
        all_headers = [
            'task_idx', 'nc1', 'nc2', 'nc3', 'nc4', 'isotropy', 'equinormity', 'recent_accuracy', 'reached_threshold'
        ] + approx_rank_headers + dead_neurons_headers + [
           
        ]
        #  'traj_avg_cos', 'traj_min_cos', 'traj_max_cos', 'traj_length'
        with open(independent_final_file, 'w', encoding='utf-8') as f:
            f.write(','.join(all_headers) + '\n')
    
    num_tasks = len(tasks_permutations)
    print(f"\n=== Starting Independent Training for {num_tasks} Tasks ===")
    
    for task_idx in range(num_tasks):
        print(f"\n--- Independent Training for Task {task_idx} ---")
        
        current_learner = create_model(params, input_size, 10, num_hidden_layers, params.get('num_features', 200), dev)#wang
        
        total_examples = int((task_idx + 1) * change_after * NUM_EPOCHS)
        total_iters = int(total_examples / mini_batch_size)
        rank_measure_period = 50000
        accuracies = torch.zeros(total_iters, dtype=torch.float)
        weight_mag_sum = torch.zeros((total_iters, num_hidden_layers + 1), dtype=torch.float)
        effective_ranks = torch.zeros((int(total_examples / rank_measure_period), num_hidden_layers), dtype=torch.float)
        approximate_ranks = torch.zeros((int(total_examples / rank_measure_period), num_hidden_layers), dtype=torch.float)
        approximate_ranks_abs = torch.zeros((int(total_examples / rank_measure_period), num_hidden_layers), dtype=torch.float)
        ranks = torch.zeros((int(total_examples / rank_measure_period), num_hidden_layers), dtype=torch.float)
        dead_neurons = torch.zeros((int(total_examples / rank_measure_period), num_hidden_layers), dtype=torch.float)
        
        iter_count = 0
        pixel_perm, data_perm = tasks_permutations[task_idx]
        
        # traj_recorder = TrajectoryMap(
        #     model=current_learner.net,
        #     step=params.get('traj_step', 100),
        #     save_dir=os.path.join(traj_save_dir, 'independent')
        # )
        os.makedirs(os.path.join(traj_save_dir, 'independent'), exist_ok=True)
        
        new_iter_start = iter_count
        
        x_task, y_task = generate_task_samples(
            x_original, y_original, pixel_perm, data_perm, examples_per_task, dev
        )
        dataset = TensorDataset(x_task, y_task)
        dataloader = DataLoader(dataset, batch_size=EVAL_BATCH_SIZE)
        
        task_start_approx_ranks = []
        task_start_dead_neurons = []
        # ResNet 无 predict() 接口，跳过秩/死亡神经元统计（CSV 以 0 占位）
        if params['agent'] != 'linear' and hasattr(current_learner.net, 'predict'):
            with torch.no_grad():
                new_idx = int(iter_count / rank_measure_period)
                m = current_learner.net.predict(x_task[:20000])[1]
                for rep_layer_idx in range(num_hidden_layers):
                    ranks[new_idx][rep_layer_idx], effective_ranks[new_idx][rep_layer_idx], \
                    approx_rank_val, approximate_ranks_abs[new_idx][rep_layer_idx] = \
                        compute_matrix_rank_summaries(m=m[rep_layer_idx], use_scipy=True)
                    task_start_approx_ranks.append(round(float(approx_rank_val.item()), 6))
                    neuron_activation_sums = m[rep_layer_idx].abs().sum(dim=0)
                    dead = (neuron_activation_sums < dead_neuron_threshold).sum()
                    task_start_dead_neurons.append(int(dead.item()))
                #print(f'[Independent Task {task_idx}] Initial approximate ranks: {task_start_approx_ranks}, Initial dead neurons: {task_start_dead_neurons}')
                del m
        
        #num_epochs = 10
        num_epochs = NUM_EPOCHS
        task_reached_threshold = False
        total_train_steps = change_after * num_epochs
        nc1_interval = NC1_INTERVAL   # 按 mini-batch 迭代数计，随 batch size 调整

        for epoch in range(num_epochs):
            if task_reached_threshold:
                break
            print(f"\n[Independent Task {task_idx}] Epoch {epoch+1}/{num_epochs}")
            set_lr_for_epoch(current_learner, params, epoch, num_epochs)

            epoch_permutation = np.random.permutation(examples_per_task)
            x_epoch = x_task[epoch_permutation]
            y_epoch = y_task[epoch_permutation]
            
            for start_idx in tqdm(range(0, change_after, mini_batch_size), desc=f"Independent Task {task_idx} Epoch {epoch+1} Training"):
                start_idx = start_idx % examples_per_task
                batch_x = x_epoch[start_idx: start_idx + mini_batch_size]
                batch_y = y_epoch[start_idx: start_idx + mini_batch_size]
                
                loss, network_output = current_learner.learn(batch_x.to(dev), batch_y.to(dev))
                
                if params.get('to_log', False) and params['agent'] != 'linear':
                    for idx, layer_idx in enumerate(current_learner.net.layers_to_log):
                        weight_mag_sum[iter_count][idx] = current_learner.net.layers[layer_idx].weight.data.abs().sum()
                
                with torch.no_grad():
                    accuracies[iter_count] = nll_accuracy(softmax(network_output, dim=1), batch_y).cpu()
                
                #traj_recorder.record_trajectory(current_step=iter_count, is_train=True)
                
                if (iter_count - new_iter_start + 1) % nc1_interval == 0:
                    nc1, nc2, nc3, nc3_max, nc3_min, nc4, isotropy, equinormity = NC(model=current_learner.net, data_loader=dataloader, num_classes=10)
                    # 改显存
                    torch.cuda.empty_cache()
                    gc.collect()
                    # del all_feats, all_labels, class_centers, classifier_weight  # 把 NC 里的大张量都删掉
                    # torch.cuda.empty_cache()
                    # gc.collect()
                    #learner.net.train()  # 切回训练模式
                    
                    total_correct = 0
                    total_samples = 0
                    with torch.no_grad():
                        for val_x, val_y in dataloader:
                            val_output = current_learner.net(val_x)
                            preds = torch.argmax(val_output, dim=1)
                            total_correct += (preds == val_y).sum().item()
                            total_samples += val_y.size(0)
                    full_accuracy = total_correct / total_samples
                    #0.96
                    if full_accuracy >= TASK_ACC_THRESHOLD:
                        task_reached_threshold = True
                        print(f"[Independent Task {task_idx}] Reached target accuracy {full_accuracy:.4f}, stopping early")
                    
                    #current_traj_metrics = traj_recorder.compute_trajectory_metrics()
                    
                    #print(f"\n[Independent Task {task_idx} Epoch {epoch+1} Iter {iter_count}] NC1: {nc1:.6f}, NC2: {nc2:.6f}, NC3: {nc3:.6f}, NC4: {nc4:.6f}")
                    print(f"\n[Independent Task {task_idx} Epoch {epoch+1} Iter {iter_count}] NC1: {nc1:.6f}, NC2: {nc2:.6f}, NC3: {nc3:.6f}, NC4: {nc4:.6f}, 等范数性: {equinormity:.6f}, 等角性: {isotropy:.6f}")
                    print(f"[Independent Task {task_idx} Epoch {epoch+1} Iter {iter_count}] Full Accuracy: {full_accuracy:.4f}")
                   # print(f"[Independent Task {task_idx} Epoch {epoch+1} Iter {iter_count}] Traj Metrics - Avg Cos: {current_traj_metrics['traj_avg_cos']:.6f}, Min Cos: {current_traj_metrics['traj_min_cos']:.6f}, Max Cos: {current_traj_metrics['traj_max_cos']:.6f}, Length: {current_traj_metrics['traj_length']}")
                    
                    with open(independent_intermediate_file, 'a', encoding='utf-8') as f:
                        f.write(
                            f"{task_idx},{iter_count},{nc1:.6f},{nc2:.6f},{nc3:.6f},{nc4:.6f},{isotropy:.6f},{equinormity:.6f},{full_accuracy:.6f}\n"
                            #f"{current_traj_metrics['traj_avg_cos']:.6f},{current_traj_metrics['traj_min_cos']:.6f},"
                            #f"{current_traj_metrics['traj_max_cos']:.6f},{current_traj_metrics['traj_length']:.0f}\n"
                        )
                
                del batch_x, batch_y
                iter_count += 1
                
                if iter_count >= len(accuracies) or task_reached_threshold:
                    break
        
        #task_traj_metrics = traj_recorder.compute_trajectory_metrics()
        task_total_iters = iter_count - new_iter_start
       # traj_recorder.plot_trajectory_map(task_idx=task_idx, total_iterations=task_total_iters)
        
        current_learner.net.eval()
        nc1, nc2, nc3, nc3_max, nc3_min, nc4, isotropy, equinormity = NC(model=current_learner.net, data_loader=dataloader, num_classes=10)
        # 改显存
        torch.cuda.empty_cache()
        gc.collect()
        # del all_feats, all_labels, class_centers, classifier_weight  # 把 NC 里的大张量都删掉
        # torch.cuda.empty_cache()
        # gc.collect()
        #learner.net.train()  # 切回训练模式
        current_learner.net.train()
        #print(f"[Independent Task {task_idx} End] NC1: {nc1:.6f}, NC2: {nc2:.6f}, NC3: {nc3:.6f}, NC4: {nc4:.6f}")
        print(f"[Independent Task {task_idx} End] NC1: {nc1:.6f}, NC2: {nc2:.6f}, NC3: {nc3:.6f}, NC4: {nc4:.6f}, 等范数性: {equinormity:.6f}, 等角性: {isotropy:.6f}")

        if params['agent'] != 'linear':
            if len(task_start_approx_ranks) != num_hidden_layers or len(task_start_dead_neurons) != num_hidden_layers:
                task_approx_ranks = [0.0] * num_hidden_layers
                task_dead_neurons = calculate_dead_neurons(current_learner, x_task, num_hidden_layers, dev)
            else:
                task_approx_ranks = task_start_approx_ranks
                task_dead_neurons = calculate_dead_neurons(current_learner, x_task, num_hidden_layers, dev)
        else:
            task_approx_ranks = [0.0] * num_hidden_layers
            task_dead_neurons = [0] * num_hidden_layers
        
        recent_acc = float(accuracies[new_iter_start:iter_count - 1].mean().item())
        
        with open(independent_final_file, 'a', encoding='utf-8') as f:
            data_row = [
                str(task_idx),
                f"{nc1:.6f}", f"{nc2:.6f}", f"{nc3:.6f}", f"{nc4:.6f}", f"{isotropy:.6f}", f"{equinormity:.6f}",
                f"{recent_acc:.6f}", str(task_reached_threshold)
            ] + [f"{ar}" for ar in task_approx_ranks] + [f"{dn}" for dn in task_dead_neurons] + [
                #f"{task_traj_metrics['traj_avg_cos']:.6f}",
                #f"{task_traj_metrics['traj_min_cos']:.6f}",
                #f"{task_traj_metrics['traj_max_cos']:.6f}",
                #f"{task_traj_metrics['traj_length']:.0f}"
            ]
            f.write(','.join(data_row) + '\n')
        
        model_path = os.path.join(independent_model_dir, f'independent_model_task_{task_idx}.pth')
        torch.save(current_learner.net.state_dict(), model_path)
        print(f"[Independent Task {task_idx}] Model saved to: {model_path}")
        
        release_tensor_memory(x_task, y_task, x_epoch, y_epoch)
    
    print(f"\n=== Independent Training Completed for All {num_tasks} Tasks ===")
    print(f"Intermediate results saved to: {independent_intermediate_file}")
    print(f"Final results saved to: {independent_final_file}")

def online_expr(params: dict):
    # ===== 按配置初始化 ResNet 迁移全局开关 =====
    global NET_TYPE, RESHAPE_TO_4D, EVAL_BATCH_SIZE, TASK_ACC_THRESHOLD, NC1_INTERVAL, NUM_EPOCHS
    global INPUT_CHANNELS, INPUT_H, INPUT_W, EXAMPLES_PER_TASK
    NET_TYPE = params.get('net_type', 'mlp')
    RESHAPE_TO_4D = (NET_TYPE == 'resnet')
    EVAL_BATCH_SIZE = params.get('eval_batch_size', 256 if NET_TYPE == 'resnet' else 1000)
    TASK_ACC_THRESHOLD = params.get('task_accuracy_threshold', 0.39)
    NC1_INTERVAL = params.get('nc1_interval', 60000)
    NUM_EPOCHS = params.get('num_epochs', NUM_EPOCHS)
    print(f"[Config] net_type={NET_TYPE}, eval_batch_size={EVAL_BATCH_SIZE}, "
          f"task_acc_threshold={TASK_ACC_THRESHOLD}, nc1_interval={NC1_INTERVAL}, num_epochs={NUM_EPOCHS}")

    agent_type = params['agent']
    initial_num_tasks = 0
    max_task_increase = 600
    target_accuracy = TASK_ACC_THRESHOLD

    num_tasks = params.get('num_tasks', 200)
    if 'num_examples' in params.keys() and "change_after" in params.keys():
        num_tasks = int(params["num_examples"] / params["change_after"])

    save_dir = params.get('save_dir', '30_cbp_cifar/')
    os.makedirs(save_dir, exist_ok=True)

    zero_csv_path = os.path.join(save_dir, '0.csv')
    if os.path.exists(zero_csv_path):
        os.remove(zero_csv_path)
        print(f"Deleted existing 0.csv file at: {zero_csv_path}")

    model_save_dir = os.path.join(save_dir, 'models')
    joint_results_dir = os.path.join(save_dir, 'joint_results')
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(joint_results_dir, exist_ok=True)

    step_size = params['step_size']
    opt = params['opt']
    weight_decay = params.get('weight_decay', 0)
    use_gpu = params.get('use_gpu', 0)
    dev = 'cpu'
    if use_gpu == 1:
        dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        # if dev == torch.device("cuda"):
        #     torch.set_default_tensor_type('torch.cuda.FloatTensor')

    to_log = params.get('to_log', False)
    num_features = params.get('num_features', 200)
    change_after = params.get('change_after', 50000)
    to_perturb = params.get('to_perturb', False)
    perturb_scale = params.get('perturb_scale', 0.1)
    num_hidden_layers = params.get('num_hidden_layers', 1)

    mini_batch_size = params.get('mini_batch_size', 1)
    # ResNet 含 BatchNorm：train 模式下 batch=1 会直接报错
    # ("Expected more than 1 value per channel")，过小 batch 也会导致 BN 统计失真
    if NET_TYPE == 'resnet' and mini_batch_size < 16:
        print(f"[Warning] net_type='resnet' 时 mini_batch_size={mini_batch_size} 过小，"
              f"已强制提升到 128（BatchNorm 限制）")
        mini_batch_size = 128
    decay_rate = params.get('decay_rate', 0.99)
    maturity_threshold = params.get('mt', 100)
    util_type = params.get('util_type', 'adaptable_contribution')

    traj_step = params.get('traj_step', 100)
    traj_save_dir = os.path.join(save_dir, 'trajectory_maps')
    os.makedirs(traj_save_dir, exist_ok=True)

    nc_data_dir = params['data_dir'].rstrip('/') + '_NC/'
    os.makedirs(nc_data_dir, exist_ok=True)

    classes_per_task = 10

    # ===== 数据集选择：cifar10 或 mnist =====
    dataset_type = params.get('dataset', 'cifar10')
    if dataset_type == 'mnist':
        images_per_class = 6000
        INPUT_CHANNELS = 1
        INPUT_H = 28
        INPUT_W = 28
        EXAMPLES_PER_TASK = 60000
        print(f"[Dataset] MNIST: 1x28x28, 60000 samples, 10 classes")
    else:
        images_per_class = 5000
        INPUT_CHANNELS = 3
        INPUT_H = 32
        INPUT_W = 32
        EXAMPLES_PER_TASK = 50000
        print(f"[Dataset] CIFAR-10: 3x32x32, 50000 samples, 10 classes")

    examples_per_task = images_per_class * classes_per_task

    # 加载数据
    if dataset_type == 'mnist':
        use_normalize = params.get('mnist_normalize', NET_TYPE == 'resnet')
        x_original, y_original, x_test, y_test = load_mnist(
            device=dev if use_gpu == 1 else 'cpu',
            flatten=True,
            normalize=use_normalize
        )
    else:
        use_grayscale = params.get('cifar_grayscale', False)
        use_normalize = params.get('cifar_normalize', NET_TYPE == 'resnet')
        x_original, y_original, x_test, y_test = load_cifar10(
            device=dev if use_gpu == 1 else 'cpu',
            flatten=True,
            use_grayscale=use_grayscale,
            normalize=use_normalize
        )
    input_size = x_original.shape[1]   # 自动获取：MNIST 784, CIFAR 彩色 3072, 灰度 1024

    num_samples = len(y_original)
    task_params = []
    all_joint_results = []

    joint_results_file = os.path.join(joint_results_dir, 'joint_training_results.csv')
    with open(joint_results_file, 'w', encoding='utf-8') as f:
        f.write('num_tasks,accuracy,nc1,nc2,nc3,nc4,isotropy,equinormity,capacity_reached,joint_reached_threshold\n')

    max_possible_tasks = min(num_tasks, initial_num_tasks + max_task_increase)

    current_learner = None
    iter_count = 0
    network_capacity = 0
    task_reached_threshold_list = []

    # ====================== 关键：先生成所有任务排列 ======================
    print(f"\n=== Generating permutations for {max_possible_tasks} tasks ===")
    tasks_permutations = generate_all_task_permutations(
        num_tasks=max_possible_tasks,
        input_size=input_size,
        num_samples=num_samples
    )

    # ====================== 顺序 1：先跑联合训练（你想要的） ======================
    print("\n" + "="*80)
    print("                  1. Start Joint Training First")
    print("="*80)

    # joint_learner, joint_acc, nc1, nc2, nc3, nc4, joint_reached_threshold = train_joint_model(
    #     params, tasks_permutations, x_original, y_original, joint_results_dir, dev
    # )

    joint_learner, joint_acc, nc1, nc2, nc3, nc4, isotropy, equinormity, joint_reached_threshold = train_joint_model(
    params, tasks_permutations, x_original, y_original, joint_results_dir, dev
    )

    joint_model_path = os.path.join(joint_results_dir, f'joint_model_{len(tasks_permutations)}_tasks.pth')
    torch.save(joint_learner.net.state_dict(), joint_model_path)
    save_model_params_to_csv(joint_learner.net, joint_results_dir, len(tasks_permutations))
    test_joint_model_per_task(joint_learner, tasks_permutations, x_original, y_original, joint_results_dir, len(tasks_permutations), dev)

    capacity_reached = not joint_reached_threshold
    # all_joint_results.append({
    #     'num_tasks': len(tasks_permutations),
    #     'accuracy': joint_acc,
    #     'nc1': nc1, 'nc2': nc2, 'nc3': nc3, 'nc4': nc4,
    #     'capacity_reached': capacity_reached,
    #     'joint_reached_threshold': joint_reached_threshold
    # })

    all_joint_results.append({
        'num_tasks': len(tasks_permutations),
        'accuracy': joint_acc,
        'nc1': nc1, 'nc2': nc2, 'nc3': nc3, 'nc4': nc4,
        'isotropy': isotropy,
        'equinormity': equinormity,
        'capacity_reached': capacity_reached,
        'joint_reached_threshold': joint_reached_threshold
    })


    with open(joint_results_file, 'a', encoding='utf-8') as f:
        f.write(f"{len(tasks_permutations)},{joint_acc:.6f},{nc1:.6f},{nc2:.6f},{nc3:.6f},{nc4:.6f},{isotropy:.6f},{equinormity:.6f},"
                f"{capacity_reached},{joint_reached_threshold}\n")

    if capacity_reached:
        network_capacity = len(tasks_permutations) - 1
        print(f"\nNetwork Capacity Reached! Max tasks n = {network_capacity}")
    else:
        print(f"Joint Training Success")

    if network_capacity == 0 and len(task_params) > 0:
        network_capacity = len(task_params)

    # ====================== 顺序 2：再跑持续学习 ======================
    print("\n" + "="*80)
    print("                  2. Start Continuous / Sequential Training")
    print("="*80)

    # 准备 plasticity 文件
    plasticity_file = os.path.join(save_dir, 'plasticity_metrics.csv')
    if not os.path.exists(plasticity_file):
        with open(plasticity_file, 'w', encoding='utf-8') as f:
            f.write('task_idx,loss_pre,loss_post,delta_norm_total,P_n\n')
    skip_threshold = params.get('skip_threshold', 0.01)   # 可从配置读取，默认0.01

    for task_idx in range(max_possible_tasks):
        print(f"\n=== Starting Task {task_idx} ===")

        total_examples = int((task_idx + 1) * change_after * NUM_EPOCHS)
        total_iters = int(total_examples / mini_batch_size)
        rank_measure_period = 50000
        accuracies = torch.zeros(total_iters, dtype=torch.float)
        weight_mag_sum = torch.zeros((total_iters, num_hidden_layers + 1), dtype=torch.float)
        effective_ranks = torch.zeros((int(total_examples / rank_measure_period), num_hidden_layers), dtype=torch.float)
        approximate_ranks = torch.zeros((int(total_examples / rank_measure_period), num_hidden_layers), dtype=torch.float)
        approximate_ranks_abs = torch.zeros((int(total_examples / rank_measure_period), num_hidden_layers), dtype=torch.float)
        ranks = torch.zeros((int(total_examples / rank_measure_period), num_hidden_layers), dtype=torch.float)
        dead_neurons = torch.zeros((int(total_examples / rank_measure_period), num_hidden_layers), dtype=torch.float)

        if current_learner is None:
            current_learner = create_model(params, input_size, classes_per_task,
                                          num_hidden_layers, num_features, dev)

        # iter_count, current_learner, _, _, task_reached_threshold = train_single_task(
        #     current_learner, x_original, y_original, task_idx, params, save_dir, traj_save_dir,
        #     num_hidden_layers, input_size, examples_per_task, change_after,
        #     mini_batch_size, rank_measure_period, iter_count, accuracies,
        #     weight_mag_sum, effective_ranks, approximate_ranks,
        #     approximate_ranks_abs, ranks, dead_neurons, dev,
        #     tasks_permutations
        # )
        iter_count, current_learner, _, _, task_reached_threshold, _ = train_single_task(
            current_learner, x_original, y_original, task_idx, params, save_dir, traj_save_dir,
            num_hidden_layers, input_size, examples_per_task, change_after,
            mini_batch_size, rank_measure_period, iter_count, accuracies,
            weight_mag_sum, effective_ranks, approximate_ranks,
            approximate_ranks_abs, ranks, dead_neurons, dev,
            tasks_permutations, plasticity_file, skip_threshold   # 新增两个参数
        )

        model_path = os.path.join(model_save_dir, f'model_task_{task_idx}.pth')
        torch.save(current_learner.net.state_dict(), model_path)
        task_params.append(model_path)
        task_reached_threshold_list.append(task_reached_threshold)

        # 前向测试
        forward_test(
            learner=current_learner,
            tasks_permutations=tasks_permutations[:task_idx+1],
            x_original=x_original,
            y_original=y_original,
            current_task_idx=task_idx,
            save_dir=save_dir,
            dev=dev
        )

    # ====================== 顺序 3：最后跑独立训练 ======================
    print("\n" + "="*80)
    print("                  3. Start Independent Training")
    print("="*80)

    train_independent_tasks(
        params=params,
        tasks_permutations=tasks_permutations,
        x_original=x_original,
        y_original=y_original,
        save_dir=save_dir,
        traj_save_dir=traj_save_dir,
        num_hidden_layers=num_hidden_layers,
        input_size=input_size,
        examples_per_task=examples_per_task,
        change_after=change_after,
        mini_batch_size=mini_batch_size,
        dev=dev
    )

    # 保存容量结果
    capacity_file = os.path.join(save_dir, 'network_capacity.txt')
    with open(capacity_file, 'w', encoding='utf-8') as f:
        f.write(f"Maximum learnable tasks (n): {network_capacity}\n")
        f.write(f"Target accuracy threshold: {target_accuracy}\n")
        f.write(f"Training epochs limit: {NUM_EPOCHS}\n")

    print(f"\nExperiment Complete! Network Capacity: {network_capacity}")

def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_cfg = os.path.join(script_dir, 'cfg', 'resnet_mnist.json')
    parser.add_argument('-c', help="Path to the config file for the experiment",
                        type=str, default=default_cfg)
    parser.add_argument('--change_after', type=int, default=None,
                        help="(optional) override change_after from config")
    args = parser.parse_args(arguments)
    cfg_file = args.c
    with open(cfg_file, 'r', encoding='utf-8') as f:
        params = json.load(f)
    if args.change_after is not None:
        params['change_after'] = int(args.change_after)
    online_expr(params)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))