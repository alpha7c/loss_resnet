import os
import torch
import pickle
import numpy as np
import pandas as pd
from typing import Optional
from model_utils import build_model
from lop.nets.deep_ffnn import DeepFFNN
from lop.nets.linear import MyLinear


def append_log_line(output_file: str, text_line: str):
    
    df_line = pd.DataFrame([[text_line]], columns=["log"])
    if os.path.exists(output_file):
        if output_file.endswith('.csv'):
            df_line.to_csv(output_file, mode='a', header=False, index=False)
        else:
            with pd.ExcelWriter(output_file, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
                df_line.to_excel(writer, header=False, index=False)
    else:
        if output_file.endswith('.csv'):
            df_line.to_csv(output_file, index=False)
        else:
            df_line.to_excel(output_file, index=False, engine='openpyxl')


def append_txt_line(txt_file: str, text_line: str):
    
    path = os.path.join(os.getcwd(), txt_file)
    line = text_line if text_line.endswith('\n') else text_line + '\n'
    with open(path, 'a', encoding='utf-8') as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())



def evaluate_saved_tasks(
    n: int,  
    samples_per_class: int,
    output_file: str,
    global_pixel_perms,
    global_classifier_weights,
    mnist_data_path: str,
    train_config_path: str = 'train_config.pkl',
    net=None,
    txt_file: Optional[str] = 'recordOfAccuracyCatastrophicLosses.txt'
):
    if net is None:
        raise ValueError("evaluate_saved_tasks  net。")
    
    
    if len(global_pixel_perms) < n  or len(global_classifier_weights) < n :
        raise ValueError(
           
        )
    
   
    with open(mnist_data_path, 'rb') as f:
        x_all, y_all, _, _ = pickle.load(f)
    with open(train_config_path, 'rb') as f:
        train_config = pickle.load(f)
    num_classes = train_config['classes_per_task']
    device = next(net.parameters()).device

    
    if isinstance(net, MyLinear):
        orig_w = net.weight.data.detach().clone()
        orig_b = net.bias.data.detach().clone() if net.bias is not None else None
    else:
        last = net.layers[-1]
        orig_w = last.weight.data.detach().clone()
        orig_b = last.bias.data.detach().clone() if last.bias is not None else None

    
    header_line = f"=====  task0~task{n-1} task{n}（now） ====="
    print(header_line)
    append_log_line(output_file, header_line)
    if txt_file:
        append_txt_line(txt_file, header_line)

    accuracies_this_eval = []  
    net.eval()
    with torch.no_grad():
        
        for task_idx in range(n-1):  
            pixel_perm = global_pixel_perms[task_idx]
            saved = global_classifier_weights[task_idx]
            
           
            if isinstance(net, MyLinear):
                net.weight.data.copy_(torch.tensor(saved['weights'], dtype=torch.float, device=device))
                if net.bias is not None and saved['biases'] is not None:
                    net.bias.data.copy_(torch.tensor(saved['biases'], dtype=torch.float, device=device))
            else:
                last = net.layers[-1]
                last.weight.data.copy_(torch.tensor(saved['weights'], dtype=torch.float, device=device))
                if last.bias is not None and saved['biases'] is not None:
                    last.bias.data.copy_(torch.tensor(saved['biases'], dtype=torch.float, device=device))
            
            
            selected_x, selected_y = [], []
            for class_idx in range(num_classes):
                class_mask = (y_all == class_idx)
                class_samples = x_all[class_mask]
                if len(class_samples) < samples_per_class:
                    raise ValueError(f"test{class_idx}sample is not enough：we need {samples_per_class}，but now {len(class_samples)}")
                np.random.seed(42 + task_idx + class_idx)
                sel = np.random.choice(len(class_samples), samples_per_class, replace=False)
                selected_x.append(class_samples[sel])
                selected_y.append(torch.full((samples_per_class,), class_idx, dtype=y_all.dtype))
            
            
            x_eval = torch.cat(selected_x, dim=0)[:, pixel_perm]
            y_eval = torch.cat(selected_y, dim=0)
            x_eval = x_eval.to(device)
            y_eval = y_eval.to(device)
            outputs = net(x_eval)
            preds = torch.argmax(outputs, dim=1)
            total_correct = (preds == y_eval).sum().item()
            acc = total_correct / (num_classes * samples_per_class)
            accuracies_this_eval.append(acc)  
            
            
            line = f"task {task_idx} accuracy: {acc:.4f}"
            print(line)
            append_log_line(output_file, line)
            if txt_file:
                append_txt_line(txt_file, line)

        
        task_idx_n = n-1  
        pixel_perm_n = global_pixel_perms[task_idx_n]
        saved_n = global_classifier_weights[task_idx_n]
        
        
        if isinstance(net, MyLinear):
            net.weight.data.copy_(torch.tensor(saved_n['weights'], dtype=torch.float, device=device))
            if net.bias is not None and saved_n['biases'] is not None:
                net.bias.data.copy_(torch.tensor(saved_n['biases'], dtype=torch.float, device=device))
        else:
            last = net.layers[-1]
            last.weight.data.copy_(torch.tensor(saved_n['weights'], dtype=torch.float, device=device))
            if last.bias is not None and saved_n['biases'] is not None:
                last.bias.data.copy_(torch.tensor(saved_n['biases'], dtype=torch.float, device=device))
        
        
        selected_x_n, selected_y_n = [], []
        for class_idx in range(num_classes):
            class_mask = (y_all == class_idx)
            class_samples = x_all[class_mask]
            if len(class_samples) < samples_per_class:
                raise ValueError(f"{class_idx}：{samples_per_class}个，{len(class_samples)}个")
            np.random.seed(42 + task_idx_n + class_idx)  
            sel_n = np.random.choice(len(class_samples), samples_per_class, replace=False)
            selected_x_n.append(class_samples[sel_n])
            selected_y_n.append(torch.full((samples_per_class,), class_idx, dtype=y_all.dtype))
        
        
        x_eval_n = torch.cat(selected_x_n, dim=0)[:, pixel_perm_n]
        y_eval_n = torch.cat(selected_y_n, dim=0)
        x_eval_n = x_eval_n.to(device)
        y_eval_n = y_eval_n.to(device)
        outputs_n = net(x_eval_n)
        preds_n = torch.argmax(outputs_n, dim=1)
        total_correct_n = (preds_n == y_eval_n).sum().item()
        acc_n = total_correct_n / (num_classes * samples_per_class)
        
        
        line_n = f"【】task {task_idx_n} accuracy: {acc_n:.4f}"
        print(line_n)
        append_log_line(output_file, line_n)
        if txt_file:
            append_txt_line(txt_file, line_n)

    
    avg_acc = float(sum(accuracies_this_eval) / len(accuracies_this_eval)) if accuracies_this_eval else 0.0
    avg_line = f"task0~task{n-2} average accuracy: {avg_acc:.4f}"
    print(avg_line)
    append_log_line(output_file, avg_line)
    if txt_file:
        append_txt_line(txt_file, avg_line)

    
    with torch.no_grad():
        if isinstance(net, MyLinear):
            net.weight.data.copy_(orig_w)
            if net.bias is not None and orig_b is not None:
                net.bias.data.copy_(orig_b)
        else:
            last = net.layers[-1]
            last.weight.data.copy_(orig_w)
            if last.bias is not None and orig_b is not None:
                last.bias.data.copy_(orig_b)

    
    return accuracies_this_eval, avg_acc