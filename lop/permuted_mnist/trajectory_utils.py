import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

plt.switch_backend('Agg')  

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
            f"Parameter normalization failed! Vector length = {norm_params.norm(2)}"
        
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

def init_trajectory_recorder(model, traj_step, traj_save_dir):
    traj_recorder = TrajectoryMap(
        model=model,
        step=traj_step,
        save_dir=traj_save_dir
    )
    return traj_recorder

def compute_and_plot_trajectory(traj_recorder, task_idx, total_task_iters):
    task_traj_metrics = traj_recorder.compute_trajectory_metrics()
    traj_recorder.plot_trajectory_map(task_idx=task_idx, total_iterations=total_task_iters)
    return task_traj_metrics
    