"""
离线评估：持续学习模型在 slot-partition（累积已见任务）上的 NC 指标。

[P1 修复配套脚本]
问题背景：
    joint_results.csv 里的联合基线 NC1（旧列名 "nc1"，新列名 "nc1_merged"）是
    merged/slot 口径——所有任务的类被压成 5 个槽位超类后计算的；而 nc_metrics.csv
    里持续侧的 NC1 是 per-task 口径（当任务自己的 5 个真实类）。两者标签空间不同，
    数值不可比（merged 的 Σ_W 含槽内全部子簇散布，被结构性抬高）。

本脚本做什么：
    利用 model_parameters/ 里已保存的每个任务结束时的持续模型
    （index-{run}_epoch-{(t+1)*num_epochs_per_task}.pt，即训完 task t 后的模型），
    在「前 t+1 个任务的累积训练数据、槽位标签 0-4」上离线计算 NC1–NC4，
    与联合侧的 merged 口径完全一致，输出 continual_slot_partition_nc.csv。

    理论上（槽内子簇数随 t 单调增加 → Σ_W 单调增加）：
        持续侧 slot-NC1(t) 应随 t 单调上升，趋近联合侧的 nc1_merged（上确界）。

    可选 --include-joint：用 joint/joint_model.pth 在同样的累积子集上评估，
    得到「同数据、不同模型」的联合曲线（控制数据量效应，比水平线更严格）。

用法（在有 torch 的环境里，于仓库任意位置执行）：
    python lop/incremental_cifar/eval_slot_partition_nc.py \
        --results-dir lop/incremental_cifar/imagenetBP \
        --config      lop/incremental_cifar/imagenetBP/config_files/index-0.json \
        [--include-joint] [--max-tasks 5]

输出：
    <results-dir>/continual_slot_partition_nc.csv
"""
import argparse
import csv
import gc
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

# 保证可以从仓库根目录 import lop.*
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from lop.incremental_cifar.incremental_cifar_experiment import IncrementalCIFARExperiment  # noqa: E402


def _to_numpy(arr):
    """torch.Tensor / np.ndarray → np.ndarray（尽量零拷贝）"""
    if torch.is_tensor(arr):
        return arr.cpu().numpy()
    return np.asarray(arr)


def build_per_task_datasets(exp, training_data):
    """
    复刻 _remap_labels_to_task 的标签语义，但直接在原始数组上操作，
    避免 restore/copy 循环带来的额外内存峰值。

    返回: list[TensorDataset]，第 t 项 = 任务 t 的训练数据，标签为槽位 0-4，
          预处理与 _make_simple_dataset（即联合训练 merged loader）完全一致。
    """
    base_imgs = _to_numpy(training_data.data["data"])
    base_lbls = _to_numpy(training_data.data["labels"])   # one-hot [N, 200]

    datasets = []
    for task_id in range(exp.num_tasks):
        task_classes = exp.task_classes[task_id]
        mask = base_lbls[:, task_classes].sum(axis=1) > 0
        task_dict = {
            "data": base_imgs[mask],
            # 列切片顺序 = task_classes 顺序 = 槽位 0-4（与 _remap_labels_to_task 一致）
            "labels": base_lbls[mask][:, task_classes],
        }
        datasets.append(exp._make_simple_dataset(task_dict))
        print("  task {0}: {1} samples, slots {2}".format(
            task_id, int(mask.sum()), task_classes.tolist()))

    # 释放基数组引用（TensorDataset 已通过 _make_simple_dataset 持有自己的张量）
    del base_imgs, base_lbls
    gc.collect()
    return datasets


def find_checkpoint(results_dir, run_index, task_idx, epochs_per_task):
    """训完 task t 的模型 = index-{run}_epoch-{(t+1)*epochs_per_task}.pt"""
    name = "index-{0}_epoch-{1}.pt".format(run_index, (task_idx + 1) * epochs_per_task)
    path = os.path.join(results_dir, "model_parameters", name)
    return path if os.path.exists(path) else None


def evaluate_on_accumulated(exp, net, per_task_datasets, up_to_task, batch_size):
    """在 tasks[0..up_to_task] 的累积数据（槽位标签）上计算 NC 指标。"""
    accumulated = ConcatDataset(per_task_datasets[: up_to_task + 1])
    loader = DataLoader(accumulated, batch_size=batch_size, shuffle=False, num_workers=0)
    # _compute_nc_on_simple_loader 内部会 clear cache、eval()、num_classes=5
    nc1, nc2, nc3, nc3_max, nc3_min, nc4, iso, equi = \
        exp._compute_nc_on_simple_loader(net, loader)
    return dict(nc1=nc1, nc2=nc2, nc3=nc3, nc3_max=nc3_max, nc3_min=nc3_min,
                nc4=nc4, isotropy=iso, equinormity=equi, n_samples=len(accumulated))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=str,
                        default=os.path.join(_SCRIPT_DIR, "imagenetBP"),
                        help="已有实验结果目录（含 model_parameters/、class_order/）")
    parser.add_argument("--config", type=str, default="",
                        help="实验 config JSON（默认取 <results-dir>/config_files/index-0.json）")
    parser.add_argument("--run-index", type=int, default=0)
    parser.add_argument("--include-joint", action="store_true",
                        help="同时用 joint/joint_model.pth 在相同累积子集上评估（推荐）")
    parser.add_argument("--max-tasks", type=int, default=-1,
                        help="只评估前 N 个任务边界（调试用；-1 = 全部）")
    parser.add_argument("--output", type=str, default="",
                        help="输出 CSV 路径（默认 <results-dir>/continual_slot_partition_nc.csv）")
    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    config_path = args.config or os.path.join(results_dir, "config_files",
                                              "index-{0}.json".format(args.run_index))
    output_path = args.output or os.path.join(results_dir, "continual_slot_partition_nc.csv")

    with open(config_path, "r", encoding="utf-8") as f:
        exp_params = json.load(f)

    print("=" * 70)
    print("Slot-partition NC evaluation (offline, no training)")
    print("  results_dir : {0}".format(results_dir))
    print("  config      : {0}".format(config_path))
    print("  output      : {0}".format(output_path))
    print("=" * 70)

    # ---- 构建实验对象（仅为复用其数据管线 / 模型构建 / NC 计算，不训练） ----
    exp = IncrementalCIFARExperiment(exp_params, results_dir=results_dir,
                                     run_index=args.run_index, verbose=True)

    # ---- 用原运行保存的 class order 覆盖 __init__ 里的随机排列（关键！） ----
    class_order_path = os.path.join(results_dir, "class_order",
                                    "index-{0}.npy".format(args.run_index))
    if not os.path.exists(class_order_path):
        raise FileNotFoundError(
            "找不到 class_order：{0}\n"
            "必须使用原运行保存的类别排列，否则任务划分与已存模型不匹配。".format(class_order_path))
    exp.all_classes = np.load(class_order_path)
    exp.task_classes = [
        exp.all_classes[i * exp.num_classes_per_task:(i + 1) * exp.num_classes_per_task]
        for i in range(exp.num_tasks)
    ]
    print("[class order] 已从 {0} 恢复（{1} 类）".format(class_order_path, len(exp.all_classes)))

    # ---- 加载训练数据（train split，与原运行相同的确定性划分：每类前 50 val、后 400 train） ----
    training_data, _ = exp.get_data(train=True, validation=False)
    per_task_datasets = build_per_task_datasets(exp, training_data)
    del training_data
    gc.collect()

    # ---- joint 模型（可选） ----
    joint_state = None
    if args.include_joint:
        joint_path = os.path.join(results_dir, "joint", "joint_model.pth")
        if os.path.exists(joint_path):
            joint_state = torch.load(joint_path, map_location=exp.device)
            print("[joint] 已加载 {0}".format(joint_path))
        else:
            print("[joint][warn] 未找到 {0}，跳过联合侧评估".format(joint_path))

    n_tasks = exp.num_tasks if args.max_tasks < 0 else min(args.max_tasks, exp.num_tasks)

    rows = []
    for t in range(n_tasks):
        ckpt = find_checkpoint(results_dir, args.run_index, t, exp.num_epochs_per_task)
        if ckpt is None:
            print("[task {0}][warn] 找不到模型 checkpoint（epoch-{1}），跳过".format(
                t, (t + 1) * exp.num_epochs_per_task))
            continue

        # ---- 持续模型：训完 task t ----
        exp.net.load_state_dict(torch.load(ckpt, map_location=exp.device))
        c = evaluate_on_accumulated(exp, exp.net, per_task_datasets, t,
                                    batch_size=exp.batch_sizes["test"])
        print("[task {0:2d}] continual slot-NC: NC1={1:.4f} NC2={2:.4f} NC3={3:.4f} NC4={4:.4f} "
              "(n={5})".format(t, c["nc1"], c["nc2"], c["nc3"], c["nc4"], c["n_samples"]))

        row = {
            "task_idx": t,
            "num_seen_tasks": t + 1,
            "num_samples": c["n_samples"],
            "nc1_slot": c["nc1"], "nc2_slot": c["nc2"],
            "nc3_slot": c["nc3"], "nc3_slot_max": c["nc3_max"], "nc3_slot_min": c["nc3_min"],
            "nc4_slot": c["nc4"],
            "isotropy_slot": c["isotropy"], "equinormity_slot": c["equinormity"],
        }

        # ---- 联合模型：同样的累积子集（控制数据量效应的严格对照） ----
        if joint_state is not None:
            exp.net.load_state_dict(joint_state)
            j = evaluate_on_accumulated(exp, exp.net, per_task_datasets, t,
                                        batch_size=exp.batch_sizes["test"])
            row.update({
                "joint_nc1_slot": j["nc1"], "joint_nc2_slot": j["nc2"],
                "joint_nc3_slot": j["nc3"], "joint_nc4_slot": j["nc4"],
            })
            print("           joint     slot-NC: NC1={0:.4f} NC3={1:.4f}".format(
                j["nc1"], j["nc3"]))

        rows.append(row)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- 写 CSV ----
    if not rows:
        raise RuntimeError("没有任何可评估的任务边界（checkpoint 缺失？）")
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print("\n已保存 {0} 行 → {1}".format(len(rows), output_path))

    # ---- 汇总：与联合侧 merged 值对比（兼容新旧列名） ----
    joint_csv = os.path.join(results_dir, "joint", "joint_results.csv")
    if os.path.exists(joint_csv):
        import pandas as pd
        jf = pd.read_csv(joint_csv)
        col = "nc1_merged" if "nc1_merged" in jf.columns else "nc1"
        print("\n[check] joint_results.csv 的 {0} = {1:.4f}".format(col, jf[col].iloc[0]))
        print("[check] 本脚本最后一个任务边界的 continual nc1_slot = {0:.4f}".format(
            rows[-1]["nc1_slot"]))
        if "joint_nc1_slot" in rows[-1]:
            print("[check] 同子集 joint nc1_slot = {0:.4f}".format(rows[-1]["joint_nc1_slot"]))
        print("[check] 若最后一行 nc1_slot 明显低于 joint merged 值，检查 class_order / "
              "checkpoint 对应关系是否正确。")


if __name__ == "__main__":
    main()
