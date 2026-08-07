import torch
import numpy as np
import os
import csv
from scipy.spatial.distance import cosine


def flatten_parameters(net):
    
    params = []
    for param in net.parameters():
       
        params.append(param.data.cpu().view(-1).numpy())
    return np.concatenate(params)


def compute_angle(v1, v2):
   
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0 
    cos_sim = np.dot(v1, v2) / (norm1 * norm2)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)  
    return np.arccos(cos_sim)


class TrajectoryMetrics:
    def __init__(self, params, net, save_dir):
       
        self.sample_interval = params.get('trajectory_sample_interval', 100)  
        self.metric_dir = os.path.join(save_dir, 'trajectory_metrics')
        os.makedirs(self.metric_dir, exist_ok=True)

        
        self.param_trajectory = [flatten_parameters(net)] 
        self.sampled_iters = [0]
        self.initial_params = self.param_trajectory[0]  

    def track(self, net, current_iter):
       
        if current_iter % self.sample_interval == 0:
            current_params = flatten_parameters(net)
            self.param_trajectory.append(current_params)
            self.sampled_iters.append(current_iter)

    def compute_all_metrics(self):
        
       
        theta = np.array(self.param_trajectory)
        n, p = theta.shape  

       
        row_norms = np.linalg.norm(theta, ord=2, axis=1, keepdims=True)
        row_norms[row_norms < 1e-10] = 1e-10  
        theta_hat = theta / row_norms

       
        c_hat = np.matmul(theta_hat, theta_hat.T)

       
        mds = np.mean(c_hat)

        
        local_angles = []
        if n >= 3:  
            updates = np.diff(theta, axis=0)  
            for i in range(len(updates) - 1):
                angle = compute_angle(updates[i], updates[i + 1])
                local_angles.append(angle)

        
        global_angles = []
        if n >= 2: 
            for i in range(1, n):  
                angle = compute_angle(self.initial_params, theta[i])
                global_angles.append(angle)

        
        results = {
            'trajectory_matrix': theta,
            'normalized_matrix': theta_hat,
            'trajectory_map': c_hat,
            'mds': mds,
            'local_angles': np.array(local_angles),
            'global_angles': np.array(global_angles),
            'sampled_iters': np.array(self.sampled_iters)
        }

        
        np.savez(os.path.join(self.metric_dir, 'metrics.npz'), **results)

        
        with open(os.path.join(self.metric_dir, 'summary.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'value'])
            writer.writerow(['mean_direction_similarity', mds])
            writer.writerow(['local_angle_mean', np.mean(local_angles) if local_angles else 0])
            writer.writerow(['local_angle_std', np.std(local_angles) if local_angles else 0])
            writer.writerow(['global_angle_mean', np.mean(global_angles) if global_angles else 0])
            writer.writerow(['global_angle_std', np.std(global_angles) if global_angles else 0])
            writer.writerow(['num_parameter_points', n])
            writer.writerow(['total_parameters', p])

        return results