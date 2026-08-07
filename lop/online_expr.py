import os
import sys
import json
import torch
import argparse
import pickle
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

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

plt.switch_backend('Agg')  

dead_neuron_threshold = 1e-8    

class TrajectoryMap:
    def __init__(self, model, step=1, save_dir="figures"):
        
        self.model = model
        self.step = step
        self.save_dir = save_dir
        self.trajectory = []  
        os.makedirs(self.save_dir, exist_ok=True) 

    def record_trajectory(self, current_step, is_train=True):
        
        if not is_train or (current_step % self.step != 0):
            return
        
        params_list = []
        for param in self.model.parameters():
            flat_param = param.data.cpu().flatten()  
            params_list.append(flat_param)
        
        if not params_list:
            return
        
        flat_params = torch.cat(params_list)
        param_norm = flat_params.norm(2)  
        
       
        eps = 1e-12 
        if param_norm < eps:
            norm_params = flat_params / (param_norm + eps)
        else:
            norm_params = flat_params / param_norm  
        
        
        assert torch.isclose(norm_params.norm(2), torch.tensor(1.0), atol=1e-4), \
            f"Parameter normalization failed! Vector length = {norm_params.norm(2)}, which should be close to 1 as required by the paper"
        
        self.trajectory.append(norm_params)

    def compute_trajectory_metrics(self):
       
        if len(self.trajectory) < 2:
            return {
                "traj_avg_cos": 0.0,    
                "traj_min_cos": 0.0,    
                "traj_max_cos": 0.0,   
                "traj_length": len(self.trajectory)  
            }
        
        traj_tensor = torch.stack(self.trajectory)
        
        cos_sim_matrix = traj_tensor @ traj_tensor.T  
        
        cos_sim_matrix = torch.clamp(cos_sim_matrix, min=-1.0, max=1.0)
        
        upper_triangle = cos_sim_matrix.triu(diagonal=1)  
        valid_cos_vals = upper_triangle[upper_triangle != 0]  
        
        return {
            "traj_avg_cos": float(valid_cos_vals.mean().item()),  
            "traj_min_cos": float(valid_cos_vals.min().item()),  
            "traj_max_cos": float(valid_cos_vals.max().item()),  
            "traj_length": len(self.trajectory)                  
        }

    def plot_trajectory_map(self, task_idx, total_iterations):
        
        if len(self.trajectory) == 0:
            print(f"[Task {task_idx}] No parameter trajectory recorded, skipping plotting")
            return
        
        traj_tensor = torch.stack(self.trajectory)
        
        cos_sim_matrix = traj_tensor @ traj_tensor.T
        
        cos_sim_matrix = torch.clamp(cos_sim_matrix, min=-1.0, max=1.0)
        
        plt.figure(figsize=(10, 8))
       
        colors = [(1, 1, 0), (0, 1, 0), (0, 0, 1)]
        custom_cmap = mcolors.LinearSegmentedColormap.from_list("blu_green_yellow", colors, N=256)
        
        im = plt.imshow(cos_sim_matrix.numpy(), cmap=custom_cmap, interpolation="nearest")
        plt.colorbar(im, label="Cosine Similarity ")  
        plt.title(f"Task {task_idx} Trajectory Map (Total Iterations: {total_iterations})")
        
        total_records = len(traj_tensor)
        if total_records > 1:
            tick_positions = np.linspace(0, total_records - 1, min(21, total_records + 1), dtype=int)
            tick_labels = [f"{int(total_iterations * pos / (total_records - 1))}" for pos in tick_positions]
        else:
            tick_positions = [0]
            tick_labels = [f"{0}"]
        
        plt.xticks(tick_positions, tick_labels, rotation=45)
        plt.yticks(tick_positions, tick_labels)
        plt.tight_layout()  
        
        save_path = os.path.join(self.save_dir, f"task_{task_idx}_trajectory_map.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"[Task {task_idx}] Trajectory Map saved to: {save_path}")

# -------------------------- 新增：模型参数保存到CSV函数（仅修改此函数） --------------------------
def save_model_params_to_csv(model, save_dir, num_tasks):
    """
    适配模型层名称：in_layer.fc、layers.0/1/2、out_layer.fc，拆分参数为5个CSV文件：
    1. 输入层：in_layer.fc.weight + in_layer.fc.bias
    2. 隐藏层1：layers.0.weight + layers.0.bias
    3. 隐藏层2：layers.1.weight + layers.1.bias
    4. 隐藏层3：layers.2.weight + layers.2.bias
    5. 输出层：out_layer.fc.weight + out_layer.fc.bias
    字段包括：layer_name, param_flat_index, param_value, num_tasks
    """
    # 严格匹配模型层名称的分组配置
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
    
    # 初始化每个文件的写入句柄并写入表头
    file_handles = {}
    for group in layer_groups:
        csv_path = os.path.join(save_dir, group["filename"])
        fh = open(csv_path, 'w', encoding='utf-8')
        fh.write('layer_name,param_flat_index,param_value,num_tasks\n')
        file_handles[group["filename"]] = fh
    
    # 遍历模型参数，按层名称匹配分组并写入
    for layer_name, param in model.named_parameters():
        # 找到当前层所属的组
        target_group = None
        for group in layer_groups:
            if layer_name in group["layer_keys"]:
                target_group = group
                break
        if not target_group:
            print(f"[Warning] 未匹配到层名称：{layer_name}，跳过该参数")
            continue  # 无匹配组则跳过（避免遗漏未知层的提示）
        
        # 写入当前层的参数到对应文件
        flat_param = param.data.cpu().flatten()
        fh = file_handles[target_group["filename"]]
        for idx, val in enumerate(flat_param):
            fh.write(f"{layer_name},{idx},{val.item():.10f},{num_tasks}\n")
    
    # 关闭所有文件句柄
    for fh in file_handles.values():
        fh.close()
    
    # 打印保存完成信息
    print(f"[Joint Training] Model parameters split into 5 CSV files in: {save_dir}")
    for group in layer_groups:
        csv_path = os.path.join(save_dir, group["filename"])
        print(f"  - {csv_path}")

# -------------------------- 新增：根据打乱方式生成任务样本 --------------------------
def generate_task_samples(x_original, y_original, pixel_perm, data_perm, examples_per_task, dev):
    """
    根据保存的打乱方式生成任务样本
    :param x_original: 原始MNIST数据x
    :param y_original: 原始MNIST数据y
    :param pixel_perm: 像素打乱排列
    :param data_perm: 数据打乱排列
    :param examples_per_task: 每个任务的样本数
    :param dev: 设备
    :return: x_task, y_task
    """
    x_task = x_original[:, pixel_perm].clone()
    x_task, y_task = x_task[data_perm], y_original[data_perm].clone()
    # 确保只取6万样本
    x_task = x_task[:examples_per_task].to(dev)
    y_task = y_task[:examples_per_task].to(dev)
    return x_task, y_task

# -------------------------- 新增：释放张量内存 --------------------------
def release_tensor_memory(*tensors):
    """释放指定张量的内存"""
    for tensor in tensors:
        if tensor is not None:
            del tensor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# -------------------------- 新增：联合模型逐任务测试准确率函数（仅修改此函数） --------------------------
def test_joint_model_per_task(joint_learner, tasks_permutations, x_original, y_original, save_dir, num_tasks, dev):
    """
    对联合训练模型逐任务测试准确率和NC1-NC4（每个任务6万样本一次性输入），结果写入CSV
    适配新的打乱方式存储逻辑，临时生成每个任务的样本
    :param joint_learner: 联合训练得到的模型learner
    :param tasks_permutations: 参与联合训练的所有任务打乱方式列表 [(pixel_perm0, data_perm0), ...]
    :param x_original: 原始MNIST数据x
    :param y_original: 原始MNIST数据y
    :param save_dir: 结果保存目录
    :param num_tasks: 参与联合训练的任务总数
    :param dev: 设备（cpu/cuda）
    """
    # 确保模型处于评估模式，不更新参数
    joint_learner.net.eval()
    
    # 定义结果文件路径
    per_task_acc_file = os.path.join(save_dir, f'joint_model_per_task_accuracy_{num_tasks}_tasks.csv')
    
    # 写入表头（新增nc1-nc4字段）
    with open(per_task_acc_file, 'w', encoding='utf-8') as f:
        f.write('task_idx,accuracy,num_samples,num_tasks,nc1,nc2,nc3,nc4\n')
    
    # 逐任务测试
    examples_per_task = 60000
    print(f"\n[Joint Model Per-Task Test] Start testing {num_tasks} tasks (each with 60000 samples)...")
    for task_idx, (pixel_perm, data_perm) in enumerate(tasks_permutations):
        print(f"  Testing Task {task_idx}...")
        
        # 临时生成当前任务样本
        x_task, y_task = generate_task_samples(
            x_original, y_original, pixel_perm, data_perm, examples_per_task, dev
        )
        
        # 禁用梯度计算，不修改模型参数
        with torch.no_grad():
            # 一次性前向传播计算所有样本输出
            output = joint_learner.net(x_task)
            preds = torch.argmax(output, dim=1)
            # 计算准确率
            total_correct = (preds == y_task).sum().item()
            total_samples = y_task.size(0)
            accuracy = total_correct / total_samples
            
            # 新增：计算当前任务的NC1-NC4
            # 创建当前任务的DataLoader（NC函数要求输入DataLoader）
            task_dataset = TensorDataset(x_task, y_task)
            task_dataloader = DataLoader(task_dataset, batch_size=1000)  # batch_size仅适配显存，不影响计算结果
            nc1, nc2, nc3, nc4 = NC(model=joint_learner.net, data_loader=task_dataloader, num_classes=10)
        
        # 打印当前任务结果（新增NC1-NC4）
        print(f"  Task {task_idx} Accuracy: {accuracy:.6f} (Correct: {total_correct}/{total_samples})")
        print(f"  Task {task_idx} NC1: {nc1:.6f}, NC2: {nc2:.6f}, NC3: {nc3:.6f}, NC4: {nc4:.6f}")
        
        # 写入CSV（新增nc1-nc4字段值）
        with open(per_task_acc_file, 'a', encoding='utf-8') as f:
            f.write(f"{task_idx},{accuracy:.6f},{total_samples},{num_tasks},{nc1:.6f},{nc2:.6f},{nc3:.6f},{nc4:.6f}\n")
        
        # 释放当前任务样本内存
        release_tensor_memory(x_task, y_task)
    
    print(f"[Joint Model Per-Task Test] Results saved to: {per_task_acc_file}")
    # 恢复模型训练模式（不影响后续逻辑）
    joint_learner.net.train()

def create_model(params, input_size, classes_per_task, num_hidden_layers, num_features, dev):
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
    return learner

def train_single_task(learner, x_original, y_original, task_idx, params, save_dir, traj_save_dir, 
                     num_hidden_layers, input_size, examples_per_task, change_after, 
                     mini_batch_size, rank_measure_period, iter, accuracies, 
                     weight_mag_sum, effective_ranks, approximate_ranks, 
                     approximate_ranks_abs, ranks, dead_neurons, dev):
    
    traj_recorder = TrajectoryMap(
        model=learner.net,
        step=params.get('traj_step', 100),
        save_dir=traj_save_dir
    )
    
    new_iter_start = iter
    # 生成并保存当前任务的打乱方式
    pixel_permutation = np.random.permutation(input_size)
    data_permutation = np.random.permutation(len(y_original))  # 原始数据的全排列
    
    # 一次性生成当前任务的6万样本（任务内复用）
    x_task, y_task = generate_task_samples(
        x_original, y_original, pixel_permutation, data_permutation, examples_per_task, dev
    )
    dataset = TensorDataset(x_task, y_task)
    dataloader = DataLoader(dataset, batch_size=1000)
    
    if params['agent'] != 'linear':
        with torch.no_grad():
            new_idx = int(iter / rank_measure_period)
            m = learner.net.predict(x_task[:20000])[1]
            task_start_approx_ranks = []
            task_start_dead_neurons = []
            for rep_layer_idx in range(num_hidden_layers):
                ranks[new_idx][rep_layer_idx], effective_ranks[new_idx][rep_layer_idx], \
                approx_rank_val, approximate_ranks_abs[new_idx][rep_layer_idx] = \
                    compute_matrix_rank_summaries(m=m[rep_layer_idx], use_scipy=True)
                task_start_approx_ranks.append(round(float(approx_rank_val.item()), 6))
                neuron_activation_sums = m[rep_layer_idx].abs().sum(dim=0)
                dead = (neuron_activation_sums < dead_neuron_threshold).sum()
                task_start_dead_neurons.append(int(dead.item()))
            print(f'[Task {task_idx}] Initial approximate ranks: {task_start_approx_ranks}, Initial dead neurons: {task_start_dead_neurons}')
    
    intermediate_file = os.path.join(save_dir, '0_0.csv')
    if not os.path.exists(intermediate_file):
        intermediate_headers = [
            'task_idx', 'iter', 'nc1', 'nc2', 'nc3', 'nc4', 'full_accuracy',
            'traj_avg_cos', 'traj_min_cos', 'traj_max_cos', 'traj_length'
        ]
        with open(intermediate_file, 'w', encoding='utf-8') as f:
            f.write(','.join(intermediate_headers) + '\n')
    
    num_epochs = 10
    task_reached_threshold = False  
    total_train_steps = change_after * num_epochs
    nc1_interval = 10000  # 实验要求：每训练10000个样本记录一次（每轮6次）
    
    for epoch in range(num_epochs):
        if task_reached_threshold:
            break  
        print(f"\n[Task {task_idx}] Epoch {epoch+1}/{num_epochs}")
       
        # 同任务内不同轮次仅打乱样本顺序，不重新生成
        epoch_permutation = np.random.permutation(examples_per_task)
        x_epoch = x_task[epoch_permutation]
        y_epoch = y_task[epoch_permutation]
        
        for start_idx in tqdm(range(0, change_after, mini_batch_size), desc=f"Task {task_idx} Epoch {epoch+1} Training"):
            start_idx = start_idx % examples_per_task
            batch_x = x_epoch[start_idx: start_idx + mini_batch_size]
            batch_y = y_epoch[start_idx: start_idx + mini_batch_size]
            
            loss, network_output = learner.learn(x=batch_x, target=batch_y)
            
            if params.get('to_log', False) and params['agent'] != 'linear':
                for idx, layer_idx in enumerate(learner.net.layers_to_log):
                    weight_mag_sum[iter][idx] = learner.net.layers[layer_idx].weight.data.abs().sum()
            
            with torch.no_grad():
                accuracies[iter] = nll_accuracy(softmax(network_output, dim=1), batch_y).cpu()
            
            traj_recorder.record_trajectory(current_step=iter, is_train=True)
            
            if (iter - new_iter_start + 1) % nc1_interval == 0:
                
                nc1, nc2, nc3, nc4 = NC(model=learner.net, data_loader=dataloader, num_classes=10)
                
                total_correct = 0
                total_samples = 0
                with torch.no_grad():
                    for val_x, val_y in dataloader:
                        val_output = learner.net(val_x)
                        preds = torch.argmax(val_output, dim=1)
                        total_correct += (preds == val_y).sum().item()
                        total_samples += val_y.size(0)
                full_accuracy = total_correct / total_samples
                
                if full_accuracy >= 0.96:  # 实验要求：准确率阈值η=0.96
                    task_reached_threshold = True
                    print(f"[Task {task_idx}] Reached target accuracy {full_accuracy:.4f}, stopping early")
                
                current_traj_metrics = traj_recorder.compute_trajectory_metrics()
                
                print(f"\n[Task {task_idx} Epoch {epoch+1} Iter {iter}] NC1: {nc1:.6f}, NC2: {nc2:.6f}, NC3: {nc3:.6f}, NC4: {nc4:.6f}")
                print(f"[Task {task_idx} Epoch {epoch+1} Iter {iter}] Full Accuracy: {full_accuracy:.4f}")
                print(f"[Task {task_idx} Epoch {epoch+1} Iter {iter}] Traj Metrics - Avg Cos: {current_traj_metrics['traj_avg_cos']:.6f}, Min Cos: {current_traj_metrics['traj_min_cos']:.6f}, Max Cos: {current_traj_metrics['traj_max_cos']:.6f}, Length: {current_traj_metrics['traj_length']}")
                
                with open(intermediate_file, 'a', encoding='utf-8') as f:
                    f.write(
                        f"{task_idx},{iter},{nc1:.6f},{nc2:.6f},{nc3:.6f},{nc4:.6f},{full_accuracy:.6f},"
                        f"{current_traj_metrics['traj_avg_cos']:.6f},{current_traj_metrics['traj_min_cos']:.6f},"
                        f"{current_traj_metrics['traj_max_cos']:.6f},{current_traj_metrics['traj_length']:.0f}\n"
                    )
            
            iter += 1  
            
            if iter >= len(accuracies) or task_reached_threshold:
                break
    
    task_traj_metrics = traj_recorder.compute_trajectory_metrics()
    task_total_iters = iter - new_iter_start
    traj_recorder.plot_trajectory_map(task_idx=task_idx, total_iterations=task_total_iters)
    
    print("[DEBUG] After NC computation: model.training =", learner.net.training)
    nc1, nc2, nc3, nc4 = NC(model=learner.net, data_loader=dataloader, num_classes=10)
    print("[DEBUG] After net.train(): model.training =", learner.net.training)
    print(f"[Task {task_idx} End] NC1: {nc1:.6f}, NC2: {nc2:.6f}, NC3: {nc3:.6f}, NC4: {nc4:.6f}")
    
    if params['agent'] != 'linear':
        if len(task_start_approx_ranks) != num_hidden_layers or len(task_start_dead_neurons) != num_hidden_layers:
            task_approx_ranks = [0.0] * num_hidden_layers
            task_dead_neurons = [0] * num_hidden_layers
        else:
            task_approx_ranks = task_start_approx_ranks
            task_dead_neurons = task_start_dead_neurons
    else:
        task_approx_ranks = [0.0] * num_hidden_layers
        task_dead_neurons = [0] * num_hidden_layers
    
    recent_acc = float(accuracies[new_iter_start:iter - 1].mean().item())
    
    task_end_file = os.path.join(save_dir, '0.csv')
    if not os.path.exists(task_end_file):
        approx_rank_headers = [f"approx_rank_{i+1}" for i in range(num_hidden_layers)]
        dead_neurons_headers = [f"dead_neurons_{i+1}" for i in range(num_hidden_layers)]
        all_headers = [
            'task_idx', 'nc1', 'nc2', 'nc3', 'nc4', 'recent_accuracy', 'reached_threshold'
        ] + approx_rank_headers + dead_neurons_headers + [
            'traj_avg_cos', 'traj_min_cos', 'traj_max_cos', 'traj_length'
        ]
        with open(task_end_file, 'w', encoding='utf-8') as f:
            f.write(','.join(all_headers) + '\n')
    
    with open(task_end_file, 'a', encoding='utf-8') as f:
        data_row = [
            str(task_idx),
            f"{nc1:.6f}", f"{nc2:.6f}", f"{nc3:.6f}", f"{nc4:.6f}",
            f"{recent_acc:.6f}", str(task_reached_threshold)
        ] + [f"{ar}" for ar in task_approx_ranks] + [f"{dn}" for dn in task_dead_neurons] + [
            f"{task_traj_metrics['traj_avg_cos']:.6f}",
            f"{task_traj_metrics['traj_min_cos']:.6f}",
            f"{task_traj_metrics['traj_max_cos']:.6f}",
            f"{task_traj_metrics['traj_length']:.0f}"
        ]
        f.write(','.join(data_row) + '\n')
    
    # 返回打乱方式而非完整样本，任务结束后释放样本内存
    release_tensor_memory(x_task, y_task, x_epoch, y_epoch, batch_x, batch_y)
    return iter, learner, pixel_permutation, data_permutation, task_reached_threshold

# -------------------------- 联合训练部分修改 --------------------------
def train_joint_model(params, tasks_permutations, x_original, y_original, save_dir, dev):
    """
    适配新的打乱方式存储逻辑，临时生成所有任务样本
    :param tasks_permutations: 任务打乱方式列表 [(pixel_perm0, data_perm0), ...]
    :param x_original: 原始MNIST数据x
    :param y_original: 原始MNIST数据y
    """
    n_tasks = len(tasks_permutations)
    samples_per_task = 60000  # 每个任务固定60000样本
    total_samples_per_epoch = n_tasks * samples_per_task  # 每轮总训练样本数
    mini_batch_size = params.get('mini_batch_size', 1)
    num_epochs = 10  # 强制最多10轮
    record_interval = total_samples_per_epoch // 6  # 每轮分6次记录
    record_points = [i * record_interval for i in range(1, 7)]  # 每轮要记录的6个样本节点
    
    # 初始化联合训练中间结果文件
    joint_intermediate_file = os.path.join(save_dir, 'joint_intermediate_results.csv')
    if not os.path.exists(joint_intermediate_file):
        intermediate_headers = [
            'epoch', 'record_idx_in_epoch', 'total_trained_samples', 
            'accuracy', 'nc1', 'nc2', 'nc3', 'nc4', 'num_tasks'
        ]
        with open(joint_intermediate_file, 'w', encoding='utf-8') as f:
            f.write(','.join(intermediate_headers) + '\n')

    # 临时生成所有任务的样本并拼接成联合数据集
    joint_x_list = []
    joint_y_list = []
    examples_per_task = 60000
    for pixel_perm, data_perm in tasks_permutations:
        x_task, y_task = generate_task_samples(
            x_original, y_original, pixel_perm, data_perm, examples_per_task, dev
        )
        joint_x_list.append(x_task)
        joint_y_list.append(y_task)
    joint_x = torch.cat(joint_x_list)
    joint_y = torch.cat(joint_y_list)
    # 释放临时列表内存
    release_tensor_memory(*joint_x_list, *joint_y_list)
    
    # 全样本评估Dataloader（必须遍历所有样本，batch_size仅为显存适配）
    full_dataset = TensorDataset(joint_x, joint_y)
    full_dataloader = DataLoader(full_dataset, batch_size=1000)  # 保证覆盖所有样本
    
    # 创建联合训练模型
    input_size = joint_x.shape[1]
    classes_per_task = 10
    num_hidden_layers = params.get('num_hidden_layers', 1)
    num_features = params.get('num_features', 2000)
    learner = create_model(params, input_size, classes_per_task, num_hidden_layers, num_features, dev)
    
    joint_reached_threshold = False
    epoch_iter = 0  # 匹配独立训练的Iter编号逻辑
    total_trained_samples_global = 0  # 联合训练累计训练样本数
    
    for epoch in range(num_epochs):
        if joint_reached_threshold:
            break
        
        # 每轮打乱联合样本顺序
        epoch_perm = np.random.permutation(total_samples_per_epoch)
        x_epoch = joint_x[epoch_perm]
        y_epoch = joint_y[epoch_perm]
        
        print(f"\nJoint Training Epoch {epoch+1}/{num_epochs}")
        current_total_samples = 0  # 本轮已训练的样本数
        recorded_points = set()  # 本轮已完成的记录点
        # 进度条：与独立训练风格完全一致
        pbar = tqdm(
            range(0, total_samples_per_epoch, mini_batch_size),
            desc=f"Joint Training Epoch {epoch+1} Training"
        )
        
        for start_idx in pbar:
            # 取当前batch（处理最后一个batch不足的情况）
            batch_x = x_epoch[start_idx: start_idx + mini_batch_size]
            batch_y = y_epoch[start_idx: start_idx + mini_batch_size]
            if len(batch_x) == 0:
                continue
            
            # 训练一步
            learner.learn(x=batch_x, target=batch_y)
            epoch_iter += 1
            current_total_samples += len(batch_x)  # 更新本轮累计训练样本数
            total_trained_samples_global += len(batch_x)  # 更新全局累计训练样本数
            
            # 检查是否达到记录点（保证每轮记录6次）
            for point in record_points:
                if point not in recorded_points and current_total_samples >= point:
                    # 计算NC指标（全样本）
                    nc1, nc2, nc3, nc4 = NC(model=learner.net, data_loader=full_dataloader, num_classes=10)
                    
                    # 全样本计算准确率：必须遍历所有联合样本
                    total_correct = 0
                    total_samples_eval = 0
                    with torch.no_grad():
                        for val_x, val_y in full_dataloader:
                            val_output = learner.net(val_x)
                            preds = torch.argmax(val_output, dim=1)
                            total_correct += (preds == val_y).sum().item()
                            total_samples_eval += val_y.size(0)
                    full_accuracy = total_correct / total_samples_eval
                    
                    # 输出格式与独立训练对齐
                    print(f"\n[Joint Training Epoch {epoch+1} Iter {epoch_iter}] NC1: {nc1:.6f}, NC2: {nc2:.6f}, NC3: {nc3:.6f}, NC4: {nc4:.6f}")
                    print(f"[Joint Training Epoch {epoch+1} Iter {epoch_iter}] Full Accuracy: {full_accuracy:.4f}")
                    
                    # 计算本轮内的记录序号
                    record_idx_in_epoch = len(recorded_points) + 1
                    # 写入中间结果文件
                    with open(joint_intermediate_file, 'a', encoding='utf-8') as f:
                        f.write(
                            f"{epoch+1},{record_idx_in_epoch},{total_trained_samples_global},"
                            f"{full_accuracy:.6f},{nc1:.6f},{nc2:.6f},{nc3:.6f},{nc4:.6f},{n_tasks}\n"
                        )
                    
                    # 达到阈值则提前结束所有训练
                    if full_accuracy >= 0.96:
                        joint_reached_threshold = True
                        print(f"Joint Training Reached target accuracy {full_accuracy:.4f}, stopping early")
                        pbar.close()  # 关闭进度条
                        break
                    
                    recorded_points.add(point)  # 标记该点已记录
            if joint_reached_threshold:
                break
        
        if joint_reached_threshold:
            break
    
    # 最终全样本评估（无论是否提前结束，都输出最终结果）
    nc1, nc2, nc3, nc4 = NC(model=learner.net, data_loader=full_dataloader, num_classes=10)
    total_correct = 0
    total_samples_eval = 0
    with torch.no_grad():
        for val_x, val_y in full_dataloader:
            val_output = learner.net(val_x)
            preds = torch.argmax(val_output, dim=1)
            total_correct += (preds == val_y).sum().item()
            total_samples_eval += val_y.size(0)
    final_accuracy = total_correct / total_samples_eval
    print(f"NC1: {nc1:.6f}, NC2: {nc2:.6f}, NC3: {nc3:.6f}, NC4: {nc4:.6f}")
    
    # 释放联合样本内存
    release_tensor_memory(joint_x, joint_y, x_epoch, y_epoch, batch_x, batch_y)
    
    return learner, final_accuracy, nc1, nc2, nc3, nc4, joint_reached_threshold
# -------------------------- 联合训练修改结束 --------------------------

def weighted_fusion_model(params, task_model_paths, input_size, classes_per_task, num_hidden_layers, num_features, dev):
    """
    加权融合多个任务模型
    """
    fusion_learner = create_model(params, input_size, classes_per_task, num_hidden_layers, num_features, dev)
    fusion_params = fusion_learner.net.state_dict()
    
    # 初始化参数累加器
    param_accumulator = {key: torch.zeros_like(val) for key, val in fusion_params.items()}
    num_models = len(task_model_paths)
    
    # 累加所有模型参数
    for model_path in task_model_paths:
        model_state = torch.load(model_path, map_location=dev, weights_only=True)
        for key, val in model_state.items():
            param_accumulator[key] += val
    
    # 计算平均参数
    for key in fusion_params.keys():
        fusion_params[key] = param_accumulator[key] / num_models
    
    # 加载平均参数到融合模型
    fusion_learner.net.load_state_dict(fusion_params)
    return fusion_learner

def online_expr(params: dict):
    agent_type = params['agent']
    initial_num_tasks = 0  
    max_task_increase = 600  
    target_accuracy = 0.96  # 实验要求：准确率阈值η=0.96
    
    num_tasks = params.get('num_tasks', 200)
    if 'num_examples' in params.keys() and "change_after" in params.keys():
        num_tasks = int(params["num_examples"] / params["change_after"])
    
    # Create save directory
    save_dir = '0/'  
    os.makedirs(save_dir, exist_ok=True)  
    
    # Create subdirectories for models and joint training results
    model_save_dir = os.path.join(save_dir, 'models')
    joint_results_dir = os.path.join(save_dir, 'joint_results')
    fusion_dir = os.path.join(save_dir, 'fusion_model')  
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(joint_results_dir, exist_ok=True)
    os.makedirs(fusion_dir, exist_ok=True)
    
    step_size = params['step_size']             
    opt = params['opt']                         
    weight_decay = params.get('weight_decay', 0)                            
    use_gpu = params.get('use_gpu', 0)                                 
    dev = 'cpu'                                 
    if use_gpu == 1:
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if dev == torch.device("cuda"):    
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    to_log = params.get('to_log', False)                              
    num_features = params.get('num_features', 2000)                         
    change_after = params.get('change_after', 60000)  # 实验要求：每轮训练样本数60000                        
    to_perturb = params.get('to_perturb', False)                          
    perturb_scale = params.get('perturb_scale', 0.1)                         
    num_hidden_layers = params.get('num_hidden_layers', 1)                       
    
    mini_batch_size = params.get('mini_batch_size', 1)                         
    decay_rate = params.get('decay_rate', 0.99)                           
    maturity_threshold = params.get('mt', 100)                    
    util_type = params.get('util_type', 'adaptable_contribution')        
    
    traj_step = params.get('traj_step', 100)  
    traj_save_dir = os.path.join(save_dir, 'trajectory_maps')  
    os.makedirs(traj_save_dir, exist_ok=True) 
    
    nc_data_dir = params['data_dir'].rstrip('/') + '_NC/'  
    os.makedirs(nc_data_dir, exist_ok=True)
    
    classes_per_task = 10
    images_per_class = 6000
    input_size = 784
    examples_per_task = images_per_class * classes_per_task
    
    # 加载原始MNIST数据（只加载一次）
    with open('data/mnist_', 'rb') as f:
        x_original, y_original, _, _ = pickle.load(f)
        if use_gpu == 1:
            x_original = x_original.to(dev)
            y_original = y_original.to(dev)
    
    # Initialize tracking variables
    task_params = []
    tasks_permutations = []  # 替换原tasks_data，存储每个任务的打乱方式
    all_joint_results = []
    
    # 初始化联合训练结果文件
    joint_results_file = os.path.join(joint_results_dir, 'joint_training_results.csv')
    with open(joint_results_file, 'w', encoding='utf-8') as f:
        f.write('num_tasks,accuracy,nc1,nc2,nc3,nc4,capacity_reached,joint_reached_threshold\n')
    
    max_possible_tasks = min(num_tasks, initial_num_tasks + max_task_increase)  # 保留最大任务上限变量
    
    current_learner = None
    iter_count = 0
    network_capacity = 0  
    
    for task_idx in range(max_possible_tasks):
        print(f"\n=== Starting Task {task_idx} (Current Total Tasks: {len(task_params)+1}) ===")
        
        # 初始化跟踪变量
        total_examples = int((len(task_params)+1) * change_after * 10)  
        total_iters = int(total_examples / mini_batch_size)
        rank_measure_period = 60000
        accuracies = torch.zeros(total_iters, dtype=torch.float)
        weight_mag_sum = torch.zeros((total_iters, num_hidden_layers + 1), dtype=torch.float)
        effective_ranks = torch.zeros((int(total_examples / rank_measure_period), num_hidden_layers), dtype=torch.float)
        approximate_ranks = torch.zeros((int(total_examples / rank_measure_period), num_hidden_layers), dtype=torch.float)
        approximate_ranks_abs = torch.zeros((int(total_examples / rank_measure_period), num_hidden_layers), dtype=torch.float)
        ranks = torch.zeros((int(total_examples / rank_measure_period), num_hidden_layers), dtype=torch.float)
        dead_neurons = torch.zeros((int(total_examples / rank_measure_period), num_hidden_layers), dtype=torch.float)
        
        # 创建/复用模型
        if current_learner is None:
            current_learner = create_model(params, input_size, classes_per_task, 
                                          num_hidden_layers, num_features, dev)
        else:
            # Continue training with previous model
            pass
        
        # 单任务训练
        print(f"\n--- Continuous Training for Task {task_idx} ---")
        iter_count, current_learner, pixel_perm, data_perm, task_reached_threshold = train_single_task(
            current_learner, x_original, y_original, task_idx, params, save_dir, traj_save_dir,
            num_hidden_layers, input_size, examples_per_task, change_after,
            mini_batch_size, rank_measure_period, iter_count, accuracies,
            weight_mag_sum, effective_ranks, approximate_ranks,
            approximate_ranks_abs, ranks, dead_neurons, dev
        )
        
        # 保存模型和打乱方式
        model_path = os.path.join(model_save_dir, f'model_task_{task_idx}.pth')
        torch.save(current_learner.net.state_dict(), model_path)
        task_params.append(model_path)
        tasks_permutations.append((pixel_perm, data_perm))  # 存储打乱方式而非完整样本
        current_num_tasks = len(task_params)
        
        # 判断是否需要联合训练
        if not task_reached_threshold or (task_idx == max_possible_tasks - 1):
            print(f"\n--- Task {task_idx} Continuous Training Failed (no threshold reached), start Joint Training ---")
            
            # 联合训练（传入打乱方式而非完整样本）
            joint_learner, joint_acc, nc1, nc2, nc3, nc4, joint_reached_threshold = train_joint_model(
                params, tasks_permutations, x_original, y_original, joint_results_dir, dev
            )
            
            # 保存联合模型
            joint_model_path = os.path.join(joint_results_dir, f'joint_model_{current_num_tasks}_tasks.pth')
            torch.save(joint_learner.net.state_dict(), joint_model_path)
            
            # 保存模型参数到CSV
            save_model_params_to_csv(joint_learner.net, joint_results_dir, current_num_tasks)
            
            # 测试联合模型（适配打乱方式逻辑）
            test_joint_model_per_task(joint_learner, tasks_permutations, x_original, y_original, joint_results_dir, current_num_tasks, dev)
            
            capacity_reached = not joint_reached_threshold
            all_joint_results.append({
                'num_tasks': current_num_tasks,
                'accuracy': joint_acc,
                'nc1': nc1, 'nc2': nc2, 'nc3': nc3, 'nc4': nc4,
                'capacity_reached': capacity_reached,
                'joint_reached_threshold': joint_reached_threshold
            })
            
            # 写入联合训练结果
            with open(joint_results_file, 'a', encoding='utf-8') as f:
                f.write(f"{current_num_tasks},{joint_acc:.6f},{nc1:.6f},{nc2:.6f},{nc3:.6f},{nc4:.6f},"
                        f"{capacity_reached},{joint_reached_threshold}\n")
            
            print(f"[Joint Training Results] Tasks: {current_num_tasks}, Accuracy: {joint_acc:.4f}, "
                  f"Capacity Reached: {capacity_reached}")
            
            # 判断是否达到网络容量
            if capacity_reached:
                network_capacity = current_num_tasks - 1  
                print(f"\nNetwork Capacity Reached! Maximum learnable tasks n = {network_capacity}")
                break
            else:
                current_learner = joint_learner
                print(f"Joint Training Success, continue to next task")
        else:
            print(f"Task {task_idx} Continuous Training Success, continue to next task")
            # 写入空结果
            with open(joint_results_file, 'a', encoding='utf-8') as f:
                f.write(f"{current_num_tasks},0.0,0.0,0.0,0.0,0.0,False,True\n")
    
    # 确定网络容量
    if network_capacity == 0 and len(task_params) > 0:
        network_capacity = len(task_params)
        print(f"\nAll tasks completed, Network Capacity n = {network_capacity}")
    
    # 保存容量结果
    capacity_file = os.path.join(save_dir, 'network_capacity.txt')
    with open(capacity_file, 'w', encoding='utf-8') as f:
        f.write(f"Maximum learnable tasks (n): {network_capacity}\n")
        f.write(f"Target accuracy threshold: {target_accuracy}\n")
        f.write(f"Training epochs limit: 10\n")
    
    # 模型融合（适配新的样本生成逻辑）
    if network_capacity > 0:
        print(f"\n=== Start Parameter Weighted Fusion for Tasks 0~{network_capacity-1} ===")
        
        fusion_task_paths = task_params[:network_capacity]
        
        # 创建融合模型
        fusion_learner = weighted_fusion_model(
            params, fusion_task_paths, input_size, classes_per_task,
            num_hidden_layers, num_features, dev
        )
        
        # 保存融合模型
        fusion_model_path = os.path.join(fusion_dir, f'fusion_model_n_{network_capacity}.pth')
        torch.save(fusion_learner.net.state_dict(), fusion_model_path)
        
        # 生成融合用的样本
        fusion_x_list = []
        fusion_y_list = []
        for pixel_perm, data_perm in tasks_permutations[:network_capacity]:
            x_task, y_task = generate_task_samples(
                x_original, y_original, pixel_perm, data_perm, examples_per_task, dev
            )
            fusion_x_list.append(x_task)
            fusion_y_list.append(y_task)
        fusion_x = torch.cat(fusion_x_list)
        fusion_y = torch.cat(fusion_y_list)
        # 释放临时列表内存
        release_tensor_memory(*fusion_x_list, *fusion_y_list)
        
        # 评估融合模型
        fusion_dataset = TensorDataset(fusion_x, fusion_y)
        fusion_dataloader = DataLoader(fusion_dataset, batch_size=1000)
        
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for val_x, val_y in fusion_dataloader:
                val_output = fusion_learner.net(val_x)
                preds = torch.argmax(val_output, dim=1)
                total_correct += (preds == val_y).sum().item()
                total_samples += val_y.size(0)
        fusion_acc = total_correct / total_samples
        
        # 计算NC指标
        nc1, nc2, nc3, nc4 = NC(model=fusion_learner.net, data_loader=fusion_dataloader, num_classes=10)
        
        # 保存融合结果
        fusion_result_file = os.path.join(fusion_dir, 'fusion_model_results.csv')
        with open(fusion_result_file, 'w', encoding='utf-8') as f:
            f.write('network_capacity,fusion_accuracy,nc1,nc2,nc3,nc4\n')
            f.write(f"{network_capacity},{fusion_acc:.6f},{nc1:.6f},{nc2:.6f},{nc3:.6f},{nc4:.6f}\n")
        
        print(f"[Fusion Model Results] Accuracy: {fusion_acc:.4f}, NC1: {nc1:.6f},NC2: {nc2:.6f},NC3: {nc3:.6f},NC4: {nc4:.6f}")
        print(f"Fusion Model Saved to: {fusion_model_path}")
        
        # 释放融合样本内存
        release_tensor_memory(fusion_x, fusion_y)
    
    print(f"\nExperiment Complete! Network Capacity: {network_capacity}")

def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', help="Path to the config file for the experiment",
                        type=str, default='temp_cfg/0.json')
    parser.add_argument('--change_after', type=int, default=None,
                        help="(optional) override change_after from config")
    args = parser.parse_args(arguments)
    cfg_file = args.c
    with open(cfg_file, 'r') as f:
        params = json.load(f)
    if args.change_after is not None:
        params['change_after'] = int(args.change_after)
    online_expr(params)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))