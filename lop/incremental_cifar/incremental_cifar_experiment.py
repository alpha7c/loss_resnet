# built-in libraries
import time
import os
import pickle
from copy import deepcopy
import json
import argparse
import csv
import math
from functools import partialmethod

# third party libraries
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import numpy as np
from torchvision import transforms

# from ml project manager
from mlproj_manager.problems import CifarDataSet
from mlproj_manager.experiments import Experiment
from mlproj_manager.util import turn_off_debugging_processes, get_random_seeds, access_dict
from mlproj_manager.util.data_preprocessing_and_transformations import ToTensor, Normalize, RandomCrop, RandomHorizontalFlip, RandomRotator
from mlproj_manager.file_management.file_and_directory_management import store_object_with_several_attempts

from lop.nets.torchvision_modified_resnet import build_resnet18, kaiming_init_resnet_module
from lop.incremental_cifar.learners import build_learner
from lop.neural_collapse_original import NC1, NC4, _get_feature_means, _get_classifier_weights
from lop.utils.neural_collapse import NC3 as NC3_full  # 返回 (nc3_mean, nc3_max, nc3_min)
from lop.utils.neural_collapse import NC2 as NC2_full  # 返回 (nc2, equinorm, equiangular)
from lop.utils.neural_collapse import clear_feature_means_cache


def subsample_cifar_data_set(sub_sample_indices, cifar_data: CifarDataSet):
    """
    Sub-samples the CIFAR 100 data set according to the given indices
    :param sub_sample_indices: array of indices in the same format as the cifar data set (numpy or torch)
    :param cifar_data: cifar data to be sub-sampled
    :return: None, but modifies the given cifar_dataset
    """

    cifar_data.data["data"] = cifar_data.data["data"][sub_sample_indices.numpy()]       # .numpy wasn't necessary with torch 2.0
    cifar_data.data["labels"] = cifar_data.data["labels"][sub_sample_indices.numpy()]
    cifar_data.integer_labels = torch.tensor(cifar_data.integer_labels)[sub_sample_indices.numpy()].tolist()
    cifar_data.current_data = cifar_data.partition_data()


class NCDataLoaderWrapper(torch.utils.data.Dataset):
    """
    Wraps a CifarDataSet (which yields dicts with "image"/"label" keys)
    into a Dataset that yields (inputs, targets) tuples with integer labels,
    compatible with the NC metric functions in lop.neural_collapse_original.
    """

    def __init__(self, cifar_dataset):
        self.cifar_dataset = cifar_dataset

    def __getitem__(self, idx):
        sample = self.cifar_dataset[idx]
        image = sample["image"]
        label = sample["label"]
        # convert one-hot label to integer class index
        if label.ndim >= 1:
            label = torch.argmax(label)
        return image, label

    def __len__(self):
        return len(self.cifar_dataset)


class IncrementalCIFARExperiment(Experiment):

    def __init__(self, exp_params: dict, results_dir: str, run_index: int, verbose=True):
        super().__init__(exp_params, results_dir, run_index, verbose)

        # set debugging options for pytorch
        debug = access_dict(exp_params, key="debug", default=True, val_type=bool)
        turn_off_debugging_processes(debug)

        # define torch device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # disable tqdm if verbose is enabled
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=self.verbose)

        """ For reproducibility """
        random_seeds = get_random_seeds()
        self.random_seed = random_seeds[self.run_index]
        torch.random.manual_seed(self.random_seed)
        torch.cuda.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        """ Experiment parameters """
        self.data_path = exp_params["data_path"]
        self.num_workers = access_dict(exp_params, key="num_workers", default=1, val_type=int)  # set to 1 when using cpu

        # optimization parameters
        self.stepsize = exp_params["stepsize"]
        self.weight_decay = exp_params["weight_decay"]
        self.momentum = exp_params["momentum"]

        # --- algorithm selection ---
        # "bp": standard SGD, "cbp": continual backprop, "ewc": elastic weight consolidation
        self.agent = access_dict(exp_params, "agent", default="bp", val_type=str,
                                 choices=["bp", "cbp", "ewc"])
        self._print("\tUsing agent: {0}".format(self.agent))

        # network resetting parameters
        self.reset_head = access_dict(exp_params, "reset_head", default=False, val_type=bool)
        self.reset_network = access_dict(exp_params, "reset_network", default=False, val_type=bool)
        if self.reset_head and self.reset_network:
            print(Warning("Resetting the whole network supersedes resetting the head of the network. There's no need to set both to True."))
        self.early_stopping = access_dict(exp_params, "early_stopping", default=False, val_type=bool)

        """ Training constants """
        self.num_tasks = access_dict(exp_params, "num_tasks", default=20, val_type=int)
        self.num_classes_per_task = 5          # 每 task 5 类，标签 0-4
        self.num_epochs_per_task = access_dict(exp_params, "num_epochs_per_task", default=200, val_type=int)
        self.num_epochs = self.num_tasks * self.num_epochs_per_task
        self.target_accuracy = access_dict(exp_params, "target_accuracy", default=0.96, val_type=float)
        self.enable_joint_training = access_dict(exp_params, "enable_joint_training", default=True, val_type=bool)
        self.enable_independent_training = access_dict(exp_params, "enable_independent_training", default=True, val_type=bool)
        self.batch_sizes = {"train": 90, "test": 100, "validation":50}
        self.num_classes = 100                 # CIFAR-100 原始类别数
        self.image_dims = (32, 32, 3)
        self.num_images_per_class = 450

        """ Network set up """
        # initialize network with 5-class output (shared across all tasks)
        self.net = build_resnet18(num_classes=self.num_classes_per_task, norm_layer=torch.nn.BatchNorm2d)
        self.net.apply(kaiming_init_resnet_module)

        # initialize optimizer
        self.optim = torch.optim.SGD(self.net.parameters(), lr=self.stepsize, momentum=self.momentum,
                                     weight_decay=self.weight_decay)

        # define loss function
        self.loss = torch.nn.CrossEntropyLoss(reduction="mean")

        # move network to device
        self.net.to(self.device)
        self.current_epoch = 0

        # build learner (encapsulates algorithm-specific forward/backward/update logic)
        self.learner = build_learner(
            agent=self.agent,
            net=self.net,
            optim=self.optim,
            loss_fn=self.loss,
            device=self.device,
            # CBP kwargs
            replacement_rate=access_dict(exp_params, "replacement_rate", default=1e-4, val_type=float),
            maturity_threshold=access_dict(exp_params, "maturity_threshold", default=100, val_type=int),
            util_type=access_dict(exp_params, "utility_function", default="weight", val_type=str),
            # EWC kwargs
            ewc_lambda=access_dict(exp_params, "ewc_lambda", default=5000.0, val_type=float),
            fisher_sample_size=access_dict(exp_params, "ewc_fisher_sample_size", default=2000, val_type=int),
        )

        """ For data partitioning """
        self.all_classes = np.random.permutation(self.num_classes)
        self.best_accuracy = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        self.best_accuracy_model_parameters = {}

        """ For storing raw data copies (used for task switching with label remapping) """
        self.base_data = {}  # keys: "train", "test", "val" → dict with "data", "labels", "integer_labels"
        self.current_task = 0  # current task index (0..19)

        """ For plasticity measurement (pre- and post-first-epoch of each task) """
        self.plasticity_loss_pre = None       # loss on task data before first epoch
        self.plasticity_params_before = None  # model state_dict before first epoch

        """ For per-task evaluation """
        self.per_task_test_data = []  # list of dicts, one per task: {"data": np.array, "labels": np.array}

        """ For creating experiment checkpoints """
        self.experiment_checkpoints_dir_path = os.path.join(self.results_dir, "experiment_checkpoints")
        self.checkpoint_identifier_name = "current_epoch"
        self.checkpoint_save_frequency = self.num_epochs_per_task  # save every time a new task is started
        self.delete_old_checkpoints = True

        """ For summaries """
        self.running_avg_window = 25
        self.current_running_avg_step, self.running_loss, self.running_accuracy = (0, 0.0, 0.0)
        self._initialize_summaries()

    # ------------------------------ Methods for initializing the experiment ------------------------------#
    def _initialize_summaries(self):
        """
        Initializes the summaries for the experiment
        """
        number_of_tasks = np.arange(self.num_epochs // self.num_epochs_per_task) + 1
        number_of_image_per_task = self.num_images_per_class * self.num_classes_per_task
        bin_size = (self.running_avg_window * self.batch_sizes["train"])
        total_checkpoints = int(np.sum(number_of_tasks * self.num_epochs_per_task * number_of_image_per_task // bin_size))

        train_prototype_array = torch.zeros(total_checkpoints, device=self.device, dtype=torch.float32)
        self.results_dict["train_loss_per_checkpoint"] = torch.zeros_like(train_prototype_array)
        self.results_dict["train_accuracy_per_checkpoint"] = torch.zeros_like(train_prototype_array)

        prototype_array = torch.zeros(self.num_epochs, device=self.device, dtype=torch.float32)
        self.results_dict["epoch_runtime"] = torch.zeros_like(prototype_array)
        self.results_dict["train_accuracy_per_epoch"] = torch.zeros_like(prototype_array)
        # test and validation summaries
        for set_type in ["test", "validation"]:
            self.results_dict[set_type + "_loss_per_epoch"] = torch.zeros_like(prototype_array)
            self.results_dict[set_type + "_accuracy_per_epoch"] = torch.zeros_like(prototype_array)
            self.results_dict[set_type + "_evaluation_runtime"] = torch.zeros_like(prototype_array)
        self.results_dict["class_order"] = self.all_classes

        # per-task accuracy matrix: task_accuracies[t][k] = accuracy on task k after training up to task t
        self.task_accuracies = torch.zeros(self.num_tasks, self.num_tasks, device=self.device, dtype=torch.float32)

        # per-task NC metrics: nc{t}_per_task[i] = metric after training task i
        nc_prototype = torch.zeros(self.num_tasks, device=self.device, dtype=torch.float32)
        for nc_name in ["nc1", "nc2", "nc3", "nc3_max", "nc3_min", "nc4", "isotropy", "equinormity"]:
            self.results_dict[nc_name + "_per_task"] = torch.zeros_like(nc_prototype)
        self.task_evaluated = torch.zeros(self.num_tasks, dtype=torch.bool)  # which tasks have been evaluated

    # -------------------------- Methods for task-continual data remapping --------------------------#
    @staticmethod
    def _store_base_data_single(cifar_data, storage_dict, key):
        """
        Store a deep copy of the raw CIFAR-100 data arrays from a CifarDataSet object.
        :param cifar_data: CifarDataSet instance containing the full data
        :param storage_dict: dict to store the copies into
        :param key: key name for this dataset (e.g. "train", "test", "val")
        """
        storage_dict[key] = {
            "data": np.array(cifar_data.data["data"]),
            "labels": np.array(cifar_data.data["labels"]),
            "integer_labels": list(cifar_data.integer_labels),
        }

    @staticmethod
    def _restore_from_base(cifar_data, base_snapshot):
        """
        Restore a CifarDataSet to its original full-data state from a stored snapshot.
        :param cifar_data: CifarDataSet instance to restore
        :param base_snapshot: dict with "data", "labels", "integer_labels" keys
        """
        cifar_data.data["data"] = base_snapshot["data"].copy()
        cifar_data.data["labels"] = base_snapshot["labels"].copy()
        cifar_data.integer_labels = list(base_snapshot["integer_labels"])
        cifar_data.current_data = cifar_data.partition_data()

    def _remap_labels_to_task(self, cifar_data, task_id):
        """
        Filter CIFAR-100 data to the 5 classes of the given task and remap one-hot labels
        from 100-dim to 5-dim (labels 0–4). Modifies cifar_data in place.

        :param cifar_data: CifarDataSet instance (should be in full-data state before calling)
        :param task_id: index of the task (0..19)
        """
        task_classes = self.all_classes[task_id * self.num_classes_per_task:
                                        (task_id + 1) * self.num_classes_per_task]

        labels_full = cifar_data.data["labels"]  # (N, 100) one-hot
        if torch.is_tensor(labels_full):
            labels_full = labels_full.cpu().numpy()

        # Find samples belonging to any of the 5 task classes
        mask = labels_full[:, task_classes].sum(axis=1) > 0

        # Filter image data
        cifar_data.data["data"] = cifar_data.data["data"][mask]
        # Remap labels: 100-dim → 5-dim one-hot
        cifar_data.data["labels"] = labels_full[mask][:, task_classes]

        # Remap integer labels to 0-4
        old_int = np.array(cifar_data.integer_labels)[mask]
        class_to_new = {int(old_cls): i for i, old_cls in enumerate(task_classes)}
        cifar_data.integer_labels = [class_to_new[int(l)] for l in old_int]

        # Update self.classes to use new 0-4 indices, otherwise partition_data()
        # will try to index the remapped 5-dim labels with original 100-class indices
        cifar_data.classes = np.arange(self.num_classes_per_task)

        # Rebuild data partitions
        cifar_data.current_data = cifar_data.partition_data()

    def _create_dataloader(self, dataset, batch_size_key):
        """
        Create a DataLoader for the given dataset.
        :param dataset: CifarDataSet instance
        :param batch_size_key: key into self.batch_sizes ("train", "test", "validation")
        """
        return DataLoader(dataset,
                          batch_size=self.batch_sizes[batch_size_key],
                          shuffle=True,
                          num_workers=self.num_workers)

    # ----------------------------- For saving and loading experiment checkpoints ----------------------------- #
    def get_experiment_checkpoint(self):
        """ Creates a dictionary with all the necessary information to pause and resume the experiment """

        partial_results = {}
        for k, v in self.results_dict.items():
            partial_results[k] = v if not isinstance(v, torch.Tensor) else v.cpu()

        checkpoint = {
            "model_weights": self.net.state_dict(),
            "optim_state": self.optim.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(),
            "epoch_number": self.current_epoch,
            "current_task": self.current_task,
            "all_classes": self.all_classes,
            "current_running_avg_step": self.current_running_avg_step,
            "partial_results": partial_results
        }

        if self.agent != "bp":
            checkpoint["learner_state"] = self.learner.get_state()
        checkpoint["task_accuracies"] = self.task_accuracies.cpu()

        return checkpoint

    def load_checkpoint_data_and_update_experiment_variables(self, file_path):
        """
        Loads the checkpoint and assigns the experiment variables the recovered values
        :param file_path: path to the experiment checkpoint
        :return: (bool) if the variables were succesfully loaded
        """

        with open(file_path, mode="rb") as experiment_checkpoint_file:
            checkpoint = pickle.load(experiment_checkpoint_file)

        self.net.load_state_dict(checkpoint["model_weights"])
        self.optim.load_state_dict(checkpoint["optim_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])
        np.random.set_state(checkpoint["numpy_rng_state"])
        self.current_epoch = checkpoint["epoch_number"]
        self.current_task = checkpoint.get("current_task", 0)
        self.all_classes = checkpoint["all_classes"]
        self.current_running_avg_step = checkpoint["current_running_avg_step"]

        partial_results = checkpoint["partial_results"]
        for k, v in self.results_dict.items():
            self.results_dict[k] = partial_results[k] if not isinstance(partial_results[k], torch.Tensor) else partial_results[k].to(self.device)

        if "learner_state" in checkpoint:
            self.learner.load_state(checkpoint["learner_state"])
        if "task_accuracies" in checkpoint:
            self.task_accuracies = checkpoint["task_accuracies"].to(self.device)

    # --------------------------------------- For storing summaries --------------------------------------- #
    def _store_training_summaries(self):
        # store train data
        self.results_dict["train_loss_per_checkpoint"][self.current_running_avg_step] += self.running_loss / self.running_avg_window
        self.results_dict["train_accuracy_per_checkpoint"][self.current_running_avg_step] += self.running_accuracy / self.running_avg_window

        self._print("\t\tOnline accuracy: {0:.2f}".format(self.running_accuracy / self.running_avg_window))
        self.running_loss *= 0.0
        self.running_accuracy *= 0.0
        self.current_running_avg_step += 1

    def _store_test_summaries(self, test_data: DataLoader, val_data: DataLoader, epoch_number: int, epoch_runtime: float):
        """ Computes test summaries and stores them in results dir """

        self.results_dict["epoch_runtime"][epoch_number] += torch.tensor(epoch_runtime, dtype=torch.float32)

        self.net.eval()
        for data_name, data_loader, compare_to_best in [("test", test_data, False), ("validation", val_data, True)]:
            # evaluate on data
            evaluation_start_time = time.perf_counter()
            loss, accuracy = self.evaluate_network(data_loader)
            evaluation_time = time.perf_counter() - evaluation_start_time

            if compare_to_best:
                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    self.best_accuracy_model_parameters = deepcopy(self.net.state_dict())

            # store summaries
            self.results_dict[data_name + "_evaluation_runtime"][epoch_number] += torch.tensor(evaluation_time, dtype=torch.float32)
            self.results_dict[data_name + "_loss_per_epoch"][epoch_number] += loss
            self.results_dict[data_name + "_accuracy_per_epoch"][epoch_number] += accuracy

            # print progress
            self._print("\t\t{0} accuracy: {1:.4f}".format(data_name, accuracy))

        self.net.train()
        self._print("\t\tEpoch run time in seconds: {0:.4f}".format(epoch_runtime))

    def evaluate_network(self, test_data: DataLoader):
        """
        Evaluates the network on the test data
        :param test_data: a pytorch DataLoader object
        :return: (torch.Tensor) test loss, (torch.Tensor) test accuracy
        """

        avg_loss = 0.0
        avg_acc = 0.0
        num_test_batches = 0
        with torch.no_grad():
            for _, sample in enumerate(test_data):
                images = sample["image"].to(self.device)
                test_labels = sample["label"].to(self.device)
                test_predictions = self.net.forward(images)

                avg_loss += self.loss(test_predictions, test_labels)
                avg_acc += torch.mean((test_predictions.argmax(axis=1) == test_labels.argmax(axis=1)).to(torch.float32))
                num_test_batches += 1

        return avg_loss / num_test_batches, avg_acc / num_test_batches

    def compute_nc_metrics(self, dataset):
        """
        Compute all four Neural Collapse metrics (NC1-NC4) for the current model.

        :param dataset: a CifarDataSet instance (e.g. test data for current task)
        :return: (nc1, nc2, nc3, nc4) tuple of floats
        """
        # wrap dataset to yield (inputs, int_label) tuples expected by NC functions
        nc_dataset = NCDataLoaderWrapper(dataset)
        nc_loader = DataLoader(nc_dataset, batch_size=self.batch_sizes["test"],
                               shuffle=False, num_workers=self.num_workers)

        clear_feature_means_cache()  # 每次 NC 计算前清除缓存，避免复用旧模型的特征
        self.net.eval()
        with torch.no_grad():
            nc1 = NC1(model=self.net, data_loader=nc_loader,
                      num_classes=self.num_classes_per_task)
            nc2, equinorm, equiangular = NC2_full(model=self.net, data_loader=nc_loader,
                      num_classes=self.num_classes_per_task, use_cache=True)
            nc3, nc3_max, nc3_min = NC3_full(model=self.net, data_loader=nc_loader,
                      num_classes=self.num_classes_per_task, use_cache=True)
            nc4 = NC4(model=self.net, data_loader=nc_loader,
                      num_classes=self.num_classes_per_task, use_cache=True)
        self.net.train()

        return nc1, nc2, nc3, nc3_max, nc3_min, nc4, equiangular, equinorm

    def _save_nc_csv(self):
        """
        Save per-task NC metrics to a CSV file for later analysis / plotting.
        File is written to: <results_dir>/nc_metrics.csv
        Also writes a 0.csv in the format expected by plot_isotropy_equinormity.py
        """
        csv_path = os.path.join(self.results_dir, "nc_metrics.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["task", "nc1", "nc2", "nc3", "nc3_max", "nc3_min", "nc4",
                             "isotropy", "equinormity"])
            for t in range(self.num_tasks):
                writer.writerow([
                    t,
                    self.results_dict["nc1_per_task"][t].item(),
                    self.results_dict["nc2_per_task"][t].item(),
                    self.results_dict["nc3_per_task"][t].item(),
                    self.results_dict["nc3_max_per_task"][t].item(),
                    self.results_dict["nc3_min_per_task"][t].item(),
                    self.results_dict["nc4_per_task"][t].item(),
                    self.results_dict["isotropy_per_task"][t].item(),
                    self.results_dict["equinormity_per_task"][t].item(),
                ])
        self._print("\tNC metrics saved to {0}".format(csv_path))

        # 同时输出 plot_isotropy_equinormity.py 所需的 0.csv（持续学习）
        continual_csv_path = os.path.join(self.results_dir, "0.csv")
        with open(continual_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["task_idx", "isotropy", "equinormity"])
            for t in range(self.num_tasks):
                writer.writerow([
                    t,
                    self.results_dict["isotropy_per_task"][t].item(),
                    self.results_dict["equinormity_per_task"][t].item(),
                ])
        self._print("\tContinual isotropy/equinormity saved to {0}".format(continual_csv_path))

    def _save_nc3_data(self, dataset):
        """
        Extract and save per-task NC3 visualization data (features, class centers,
        classifier weights) as a pickle file, compatible with visualize_nc3.py.

        File is written to: <results_dir>/nc3_saved_data/nc3_data_task_<current_task>.pkl

        :param dataset: a CifarDataSet instance for the current task
        """
        nc_loader = DataLoader(NCDataLoaderWrapper(dataset),
                               batch_size=self.batch_sizes["test"],
                               shuffle=False, num_workers=self.num_workers)

        self.net.eval()
        with torch.no_grad():
            # collect all features and labels
            all_feats = []
            all_labels = []
            for inputs, targets in nc_loader:
                inputs = inputs.to(self.device)
                feats = self.net.embed(inputs)
                all_feats.append(feats.cpu())
                all_labels.append(targets.cpu())

        all_feats = torch.cat(all_feats)
        all_labels = torch.cat(all_labels)

        # class centers and classifier weights
        mu_G, mu_c_dict = _get_feature_means(
            model=self.net, data_loader=nc_loader,
            num_classes=self.num_classes_per_task, use_cache=False
        )
        class_centers = torch.stack([mu_c_dict[c] for c in range(self.num_classes_per_task)])
        classifier_weight = _get_classifier_weights(self.net)

        # sample up to 200 features per class
        samples_per_class = 200
        sample_features_list = []
        sample_labels_list = []
        for c in range(self.num_classes_per_task):
            class_mask = (all_labels == c)
            class_feats = all_feats[class_mask][:samples_per_class]
            class_lbls = all_labels[class_mask][:samples_per_class]
            sample_features_list.append(class_feats)
            sample_labels_list.append(class_lbls)
        sample_features = torch.cat(sample_features_list).numpy()
        sample_labels = torch.cat(sample_labels_list).numpy()

        self.net.train()

        # save pickle
        save_dir = os.path.join(self.results_dir, "nc3_saved_data")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"nc3_data_task_{self.current_task}.pkl")
        save_dict = {
            "task_idx": self.current_task,
            "class_centers": class_centers.detach().cpu().numpy(),
            "classifier_weights": classifier_weight.detach().cpu().numpy(),
            "sample_features": sample_features,
            "sample_labels": sample_labels,
        }
        with open(save_path, "wb") as f:
            pickle.dump(save_dict, f)
        self._print("\tNC3 data saved to {0}".format(save_path))

    # ------------------------- Shared helper for joint / independent training ------------------------- #
    def _create_fresh_model_and_optimizer(self):
        """
        Create a fresh ResNet-18 and SGD optimizer with the same config as the continual model.
        :return: (net, optim) tuple, both on self.device
        """
        net = build_resnet18(num_classes=self.num_classes_per_task, norm_layer=torch.nn.BatchNorm2d)
        net.apply(kaiming_init_resnet_module)
        optim = torch.optim.SGD(net.parameters(), lr=self.stepsize, momentum=self.momentum,
                                weight_decay=self.weight_decay)
        return net.to(self.device), optim

    def _collect_all_tasks_data(self, training_data, val_data, test_data):
        """
        Iterate over all tasks and collect (remapped) train/val/test data for each task.
        Returns three lists of dicts: train_data_list, val_data_list, test_data_list.
        Each dict contains {"data": np.array, "labels": np.array, "integer_labels": list}.
        The original data objects are restored to base state after collection.
        """
        train_data_list, val_data_list, test_data_list = [], [], []

        for task_id in range(self.num_tasks):
            # restore and remap training data
            self._restore_from_base(training_data, self.base_data["train"])
            self._remap_labels_to_task(training_data, task_id)
            train_data_list.append({
                "data": np.array(training_data.data["data"]),
                "labels": np.array(training_data.data["labels"]),
                "integer_labels": list(training_data.integer_labels),
            })

            # restore and remap val data
            self._restore_from_base(val_data, self.base_data["val"])
            self._remap_labels_to_task(val_data, task_id)
            val_data_list.append({
                "data": np.array(val_data.data["data"]),
                "labels": np.array(val_data.data["labels"]),
                "integer_labels": list(val_data.integer_labels),
            })

            # restore and remap test data
            self._restore_from_base(test_data, self.base_data["test"])
            self._remap_labels_to_task(test_data, task_id)
            test_data_list.append({
                "data": np.array(test_data.data["data"]),
                "labels": np.array(test_data.data["labels"]),
                "integer_labels": list(test_data.integer_labels),
            })

        # restore original state
        self._restore_from_base(training_data, self.base_data["train"])
        self._restore_from_base(val_data, self.base_data["val"])
        self._restore_from_base(test_data, self.base_data["test"])

        return train_data_list, val_data_list, test_data_list

    @staticmethod
    def _make_simple_dataset(data_dict):
        """
        Create a simple torch Dataset from a dict with "data" (np array) and "labels" (np array).
        Handles HWC→CHW conversion and CIFAR-100 normalization.
        Yields (image, label) tuples with integer labels (not one-hot).
        """
        # data is raw (N, H, W, C) uint8 in [0, 255]
        images = torch.from_numpy(data_dict["data"]).float() / 255.0
        # HWC → CHW
        if images.ndim == 4 and images.shape[-1] == 3:
            images = images.permute(0, 3, 1, 2)
        # CIFAR-100 normalization
        mean = torch.tensor([0.5071, 0.4865, 0.4409]).view(1, 3, 1, 1)
        std = torch.tensor([0.2673, 0.2564, 0.2762]).view(1, 3, 1, 1)
        images = (images - mean) / std

        labels_array = data_dict["labels"]
        if labels_array.ndim > 1:
            # one-hot → integer
            labels = torch.from_numpy(np.argmax(labels_array, axis=1)).long()
        else:
            labels = torch.from_numpy(labels_array).long()
        return torch.utils.data.TensorDataset(images, labels)

    # ----------------------------- Joint training ----------------------------- #
    def train_joint_model(self, training_data, val_data, test_data):
        """
        Joint training: concatenate all tasks' data and train one model from scratch.
        Results are saved to <results_dir>/joint/
        """
        joint_dir = os.path.join(self.results_dir, "joint")
        os.makedirs(joint_dir, exist_ok=True)

        self._print("\n" + "=" * 60)
        self._print("\t=== Joint Training: all {0} tasks together ===".format(self.num_tasks))
        self._print("=" * 60)

        # collect all task data
        train_list, val_list, test_list = self._collect_all_tasks_data(
            training_data, val_data, test_data
        )

        # concatenate into one big dataset for each split
        joint_train_data = {
            "data": np.concatenate([t["data"] for t in train_list], axis=0),
            "labels": np.concatenate([t["labels"] for t in train_list], axis=0),
        }
        joint_val_data = {
            "data": np.concatenate([t["data"] for t in val_list], axis=0),
            "labels": np.concatenate([t["labels"] for t in val_list], axis=0),
        }
        joint_test_data = {
            "data": np.concatenate([t["data"] for t in test_list], axis=0),
            "labels": np.concatenate([t["labels"] for t in test_list], axis=0),
        }

        self._print("\tJoint train samples: {0}".format(len(joint_train_data["data"])))
        self._print("\tJoint test samples: {0}".format(len(joint_test_data["data"])))

        # create fresh model
        net, optim = self._create_fresh_model_and_optimizer()
        loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")

        # build learner (BP only for joint training)
        joint_learner = build_learner(
            agent="bp", net=net, optim=optim, loss_fn=loss_fn, device=self.device,
        )

        # create datasets and dataloaders
        train_dataset = self._make_simple_dataset(joint_train_data)
        val_dataset = self._make_simple_dataset(joint_val_data)
        test_dataset = self._make_simple_dataset(joint_test_data)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_sizes["train"],
                                  shuffle=True, num_workers=self.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_sizes["validation"],
                                shuffle=False, num_workers=self.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_sizes["test"],
                                 shuffle=False, num_workers=self.num_workers)

        # intermediate results file
        intermediate_file = os.path.join(joint_dir, "joint_intermediate.csv")
        with open(intermediate_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_acc", "test_loss", "test_acc",
                             "nc1", "nc2", "nc3", "nc4"])

        joint_reached_threshold = False
        best_val_acc = 0.0
        best_model_state = None

        for epoch in tqdm(range(self.num_epochs_per_task), desc="Joint Training"):
            if joint_reached_threshold:
                break

            # ---- train one epoch ----
            net.train()
            total_loss, total_correct, total_samples = 0.0, 0, 0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                # one-hot encode labels for the learner
                labels_oh = torch.nn.functional.one_hot(labels, num_classes=self.num_classes_per_task).float()

                for param in net.parameters():
                    param.grad = None

                loss, preds = joint_learner.learn(images, labels_oh)

                total_loss += loss.item() * images.size(0)
                total_correct += (preds.argmax(dim=1) == labels).sum().item()
                total_samples += images.size(0)

            train_loss = total_loss / total_samples
            train_acc = total_correct / total_samples

            # ---- evaluate ----
            net.eval()
            test_loss, test_correct, test_samples = 0.0, 0, 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    labels_oh = torch.nn.functional.one_hot(labels, num_classes=self.num_classes_per_task).float()
                    logits = net(images)
                    test_loss += loss_fn(logits, labels_oh).item() * images.size(0)
                    test_correct += (logits.argmax(dim=1) == labels).sum().item()
                    test_samples += images.size(0)
            test_loss_val = test_loss / test_samples
            test_acc = test_correct / test_samples

            # eval on val set for best model tracking
            val_correct, val_samples = 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    logits = net(images)
                    val_correct += (logits.argmax(dim=1) == labels).sum().item()
                    val_samples += images.size(0)
            val_acc = val_correct / val_samples
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = deepcopy(net.state_dict())

            self._print("\t[Joint] Epoch {0}: train_acc={1:.4f}, test_acc={2:.4f}".format(
                epoch + 1, train_acc, test_acc))

            # ---- compute NC every 10 epochs ----
            nc1, nc2, nc3, nc3_max, nc3_min, nc4 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            _iso, _equi = 0.0, 0.0
            if (epoch + 1) % 10 == 0:
                try:
                    nc1, nc2, nc3, nc3_max, nc3_min, nc4, _iso, _equi = self._compute_nc_on_simple_loader(
                        net, test_loader)
                    self._print("\t[Joint] Epoch {0}: NC1={1:.4f}, NC2={2:.4f}, NC3={3:.4f}, NC4={4:.4f}".format(
                        epoch + 1, nc1, nc2, nc3, nc4))
                except Exception as e:
                    self._print("\t[Joint] NC computation failed: {0}".format(e))

            # save intermediate
            with open(intermediate_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([epoch + 1, train_loss, train_acc, test_loss_val, test_acc,
                                 nc1, nc2, nc3, nc4])

            # early stop check (train accuracy)
            if train_acc >= self.target_accuracy:
                joint_reached_threshold = True
                self._print("\t[Joint] Target train accuracy {0:.2%} reached at epoch {1}, stopping.".format(
                    self.target_accuracy, epoch + 1))

        # restore best model
        if best_model_state is not None:
            net.load_state_dict(best_model_state)

        # final NC computation
        net.eval()
        try:
            nc1, nc2, nc3, nc3_max, nc3_min, nc4, _iso, _equi = self._compute_nc_on_simple_loader(net, test_loader)
        except Exception as e:
            self._print("\t[Joint] Final NC failed: {0}".format(e))
            nc1, nc2, nc3, nc3_max, nc3_min, nc4 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        # final accuracy
        final_correct, final_samples = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                logits = net(images)
                final_correct += (logits.argmax(dim=1) == labels).sum().item()
                final_samples += images.size(0)
        final_acc = final_correct / final_samples

        self._print("\t[Joint] Final accuracy: {0:.4f}".format(final_acc))

        # save results
        results_file = os.path.join(joint_dir, "joint_results.csv")
        with open(results_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["num_tasks", "accuracy", "nc1", "nc2", "nc3", "nc4",
                             "reached_threshold", "epochs_trained"])
            writer.writerow([self.num_tasks, final_acc, nc1, nc2, nc3, nc4,
                             joint_reached_threshold, epoch + 1])

        # save model
        torch.save(net.state_dict(), os.path.join(joint_dir, "joint_model.pth"))

        # =====================================================================
        # Per-task evaluation: evaluate joint model on each task individually
        # =====================================================================
        self._print("\n\t[Joint] Per-task evaluation on all {0} tasks...".format(self.num_tasks))
        per_task_file = os.path.join(joint_dir, "joint_per_task_results.csv")
        with open(per_task_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["task_idx", "accuracy", "num_samples", "nc1", "nc2", "nc3", "nc4",
                             "isotropy", "equinormity"])

        net.eval()
        joint_iso_equi = []  # 收集每个 task 的 (isotropy, equinormity)
        for task_id in range(self.num_tasks):
            task_test_dataset = self._make_simple_dataset(test_list[task_id])
            task_test_loader = DataLoader(task_test_dataset,
                                          batch_size=self.batch_sizes["test"],
                                          shuffle=False, num_workers=self.num_workers)

            # compute per-task accuracy
            task_correct, task_samples = 0, 0
            with torch.no_grad():
                for images, labels in task_test_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    logits = net(images)
                    task_correct += (logits.argmax(dim=1) == labels).sum().item()
                    task_samples += images.size(0)
            task_acc = task_correct / task_samples

            # compute per-task NC metrics
            try:
                tnc1, tnc2, tnc3, tnc3_max, tnc3_min, tnc4, t_iso, t_equi = self._compute_nc_on_simple_loader(net, task_test_loader)
            except Exception:
                tnc1, tnc2, tnc3, tnc3_max, tnc3_min, tnc4, t_iso, t_equi = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

            self._print("\t[Joint] Task {0}: acc={1:.4f}, NC1={2:.4f}, NC2={3:.4f}, NC3={4:.4f}, NC4={5:.4f}".format(
                task_id, task_acc, tnc1, tnc2, tnc3, tnc4))

            with open(per_task_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([task_id, task_acc, task_samples, tnc1, tnc2, tnc3, tnc4,
                                 t_iso, t_equi])

            joint_iso_equi.append((t_iso, t_equi))

        self._print("\t[Joint] Per-task results saved to {0}".format(per_task_file))

        # 输出 plot_isotropy_equinormity.py 所需的联合训练文件
        joint_plot_file = os.path.join(joint_dir,
            "joint_model_per_task_accuracy_{0}_tasks.csv".format(self.num_tasks))
        with open(joint_plot_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["task_idx", "isotropy", "equinormity"])
            for task_id in range(self.num_tasks):
                writer.writerow([
                    task_id,
                    joint_iso_equi[task_id][0],
                    joint_iso_equi[task_id][1],
                ])
        self._print("\t[Joint] Plot-compatible isotropy/equinormity saved to {0}".format(joint_plot_file))

        self._print("\t[Joint] Results saved to {0}".format(joint_dir))
        del net, optim, joint_learner
        torch.cuda.empty_cache()

    def _compute_nc_on_simple_loader(self, net, loader):
        """Compute NC metrics using a simple (image, int_label) DataLoader.
        Returns: (nc1, nc2, nc3, nc3_max, nc3_min, nc4, isotropy, equinormity)
        """
        clear_feature_means_cache()  # 清除缓存（独立训练每 task 换模型，联合训练每 task 换数据子集）
        net.eval()
        with torch.no_grad():
            nc1 = NC1(model=net, data_loader=loader, num_classes=self.num_classes_per_task)
            nc2, equinorm, equiangular = NC2_full(model=net, data_loader=loader,
                                                   num_classes=self.num_classes_per_task, use_cache=True)
            nc3, nc3_max, nc3_min = NC3_full(model=net, data_loader=loader,
                                              num_classes=self.num_classes_per_task, use_cache=True)
            nc4 = NC4(model=net, data_loader=loader, num_classes=self.num_classes_per_task, use_cache=True)
        return nc1, nc2, nc3, nc3_max, nc3_min, nc4, equiangular, equinorm

    # ----------------------------- Independent training ----------------------------- #
    def train_independent_models(self, training_data, val_data, test_data):
        """
        Independent training: train a separate model from scratch for each task.
        Results are saved to <results_dir>/independent/
        """
        indep_dir = os.path.join(self.results_dir, "independent")
        os.makedirs(indep_dir, exist_ok=True)
        models_dir = os.path.join(indep_dir, "models")
        os.makedirs(models_dir, exist_ok=True)

        self._print("\n" + "=" * 60)
        self._print("\t=== Independent Training: {0} tasks, each from scratch ===".format(self.num_tasks))
        self._print("=" * 60)

        # collect all task data
        train_list, val_list, test_list = self._collect_all_tasks_data(
            training_data, val_data, test_data
        )

        # final results file (one row per task)
        results_file = os.path.join(indep_dir, "independent_results.csv")
        with open(results_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["task_idx", "nc1", "nc2", "nc3", "nc4",
                             "isotropy", "equinormity",
                             "accuracy", "reached_threshold", "epochs_trained"])

        # intermediate file (per-epoch details)
        intermediate_file = os.path.join(indep_dir, "independent_intermediate.csv")
        with open(intermediate_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["task_idx", "epoch", "train_loss", "train_acc",
                             "test_loss", "test_acc", "nc1", "nc2", "nc3", "nc4"])

        indep_iso_equi = []  # 收集每个独立 task 的 (isotropy, equinormity)
        for task_id in range(self.num_tasks):
            self._print("\n\t--- Independent Task {0}/{1} ---".format(task_id + 1, self.num_tasks))

            # create fresh model per task
            net, optim = self._create_fresh_model_and_optimizer()
            loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
            indep_learner = build_learner(
                agent="bp", net=net, optim=optim, loss_fn=loss_fn, device=self.device,
            )

            # build datasets for this task
            train_dataset = self._make_simple_dataset(train_list[task_id])
            val_dataset = self._make_simple_dataset(val_list[task_id])
            test_dataset = self._make_simple_dataset(test_list[task_id])

            train_loader = DataLoader(train_dataset, batch_size=self.batch_sizes["train"],
                                      shuffle=True, num_workers=self.num_workers)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_sizes["validation"],
                                    shuffle=False, num_workers=self.num_workers)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_sizes["test"],
                                     shuffle=False, num_workers=self.num_workers)

            task_reached_threshold = False
            best_val_acc = 0.0
            best_model_state = None
            epochs_trained = 0

            for epoch in range(self.num_epochs_per_task):
                if task_reached_threshold:
                    break
                epochs_trained = epoch + 1

                # ---- train one epoch ----
                net.train()
                total_loss, total_correct, total_samples = 0.0, 0, 0
                for images, labels in train_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    labels_oh = torch.nn.functional.one_hot(labels, num_classes=self.num_classes_per_task).float()

                    for param in net.parameters():
                        param.grad = None

                    loss, preds = indep_learner.learn(images, labels_oh)

                    total_loss += loss.item() * images.size(0)
                    total_correct += (preds.argmax(dim=1) == labels).sum().item()
                    total_samples += images.size(0)

                train_loss = total_loss / total_samples
                train_acc = total_correct / total_samples

                # ---- evaluate ----
                net.eval()
                test_loss, test_correct, test_samples = 0.0, 0, 0
                with torch.no_grad():
                    for images, labels in test_loader:
                        images, labels = images.to(self.device), labels.to(self.device)
                        labels_oh = torch.nn.functional.one_hot(labels, num_classes=self.num_classes_per_task).float()
                        logits = net(images)
                        test_loss += loss_fn(logits, labels_oh).item() * images.size(0)
                        test_correct += (logits.argmax(dim=1) == labels).sum().item()
                        test_samples += images.size(0)
                test_loss_val = test_loss / test_samples
                test_acc = test_correct / test_samples

                # track best val model
                val_correct, val_samples = 0, 0
                with torch.no_grad():
                    for images, labels in val_loader:
                        images, labels = images.to(self.device), labels.to(self.device)
                        logits = net(images)
                        val_correct += (logits.argmax(dim=1) == labels).sum().item()
                        val_samples += images.size(0)
                val_acc = val_correct / val_samples
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = deepcopy(net.state_dict())

                # ---- NC every 10 epochs ----
                nc1, nc2, nc3, nc3_max, nc3_min, nc4 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                _iso, _equi = 0.0, 0.0
                if (epoch + 1) % 10 == 0:
                    try:
                        nc1, nc2, nc3, nc3_max, nc3_min, nc4, _iso, _equi = self._compute_nc_on_simple_loader(net, test_loader)
                    except Exception as e:
                        pass

                # save intermediate
                with open(intermediate_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([task_id, epoch + 1, train_loss, train_acc,
                                     test_loss_val, test_acc, nc1, nc2, nc3, nc4])

                # 使用 train_acc 做 early stopping
                if train_acc >= self.target_accuracy:
                    task_reached_threshold = True
                    self._print("\t[Independent Task {0}] Target train accuracy reached at epoch {1} (train_acc={2:.3f})".format(
                        task_id, epoch + 1, train_acc))

            # restore best model
            if best_model_state is not None:
                net.load_state_dict(best_model_state)

            # final NC
            net.eval()
            try:
                nc1, nc2, nc3, nc3_max, nc3_min, nc4, iso, equi = self._compute_nc_on_simple_loader(net, test_loader)
            except Exception:
                nc1, nc2, nc3, nc3_max, nc3_min, nc4, iso, equi = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

            # final accuracy
            final_correct, final_samples = 0, 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    logits = net(images)
                    final_correct += (logits.argmax(dim=1) == labels).sum().item()
                    final_samples += images.size(0)
            final_acc = final_correct / final_samples

            self._print("\t[Independent Task {0}] Final acc={1:.4f}, NC1={2:.4f}, NC2={3:.4f}, NC3={4:.4f}, NC4={5:.4f}".format(
                task_id, final_acc, nc1, nc2, nc3, nc4))

            # save task results
            with open(results_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([task_id, nc1, nc2, nc3, nc4, iso, equi, final_acc,
                                 task_reached_threshold, epochs_trained])

            indep_iso_equi.append((iso, equi))

            # save model
            torch.save(net.state_dict(), os.path.join(models_dir, "independent_model_task_{0}.pth".format(task_id)))

            del net, optim, indep_learner
            torch.cuda.empty_cache()

        # 输出 plot_isotropy_equinormity.py 所需的独立训练文件
        indep_plot_file = os.path.join(self.results_dir, "independent_final_results.csv")
        with open(indep_plot_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["task_idx", "isotropy", "equinormity"])
            for task_id in range(self.num_tasks):
                writer.writerow([
                    task_id,
                    indep_iso_equi[task_id][0],
                    indep_iso_equi[task_id][1],
                ])
        self._print("\t[Independent] Plot-compatible isotropy/equinormity saved to {0}".format(indep_plot_file))

        self._print("\t[Independent] All tasks done. Results saved to {0}".format(indep_dir))

    # ------------------------------------- For running the experiment ------------------------------------- #
    def run(self):
        # load data (full CIFAR-100, after train/val split but before any class filtering)
        training_data, training_dataloader = self.get_data(train=True, validation=False)
        val_data, val_dataloader = self.get_data(train=True, validation=True)
        test_data, test_dataloader = self.get_data(train=False)

        # store base (unfiltered) copies for task switching with label remapping
        self._store_base_data_single(training_data, self.base_data, "train")
        self._store_base_data_single(val_data, self.base_data, "val")
        self._store_base_data_single(test_data, self.base_data, "test")

        # pre-create per-task test data snapshots for multi-task evaluation
        for task_id in range(self.num_tasks):
            # create a temporary filtered copy of the test data
            task_classes = self.all_classes[task_id * self.num_classes_per_task:
                                            (task_id + 1) * self.num_classes_per_task]
            labels_full = self.base_data["test"]["labels"]
            mask = labels_full[:, task_classes].sum(axis=1) > 0
            self.per_task_test_data.append({
                "data": self.base_data["test"]["data"][mask].copy(),
                "labels": labels_full[mask][:, task_classes].copy(),
                "classes": task_classes.copy(),
            })

        # load checkpoint if one is available
        self.load_experiment_checkpoint()

        # =====================================================================
        # ① Joint Training: all tasks' data concatenated, one model from scratch
        # =====================================================================
        if self.enable_joint_training:
            self.train_joint_model(training_data, val_data, test_data)

        # =====================================================================
        # ② Continual Training: one model, sequential tasks (existing logic)
        # =====================================================================
        # set up data for task 0 (the train loop handles task switching thereafter)
        self._restore_from_base(training_data, self.base_data["train"])
        self._restore_from_base(val_data, self.base_data["val"])
        self._restore_from_base(test_data, self.base_data["test"])
        self._remap_labels_to_task(training_data, self.current_task)
        self._remap_labels_to_task(val_data, self.current_task)
        self._remap_labels_to_task(test_data, self.current_task)
        training_dataloader = self._create_dataloader(training_data, "train")
        val_dataloader = self._create_dataloader(val_data, "validation")
        test_dataloader = self._create_dataloader(test_data, "test")

        # train network
        self.train(train_dataloader=training_dataloader, test_dataloader=test_dataloader, val_dataloader=val_dataloader,
                   test_data=test_data, training_data=training_data, val_data=val_data)

        # final evaluation: after all tasks, evaluate on every task
        self._print("\n\t=== Final evaluation on all tasks ===")
        self._evaluate_all_seen_tasks(test_data)
        self.results_dict["task_accuracies"] = self.task_accuracies.cpu()
        self._print("\tFinal per-task accuracies:\n{0}".format(self.task_accuracies))
        self._save_task_accuracies_csv()
        # save NC metrics as a CSV for plotting
        self._save_nc_csv()

        # =====================================================================
        # ③ Independent Training: one fresh model per task
        # =====================================================================
        if self.enable_independent_training:
            self.train_independent_models(training_data, val_data, test_data)
        # store results using exp.store_results()

    def get_data(self, train: bool = True, validation: bool = False):
        """
        Loads the data set
        :param train: (bool) indicates whether to load the train (True) or the test (False) data
        :param validation: (bool) indicates whether to return the validation set. The validation set is made up of
                           50 examples of each class of whichever set was loaded
        :return: data set, data loader
        """

        """ Loads CIFAR data set """
        cifar_data = CifarDataSet(root_dir=self.data_path,
                                  train=train,
                                  cifar_type=100,
                                  device=None,
                                  image_normalization="max",
                                  label_preprocessing="one-hot",
                                  use_torch=True)

        mean = (0.5071, 0.4865, 0.4409)
        std = (0.2673, 0.2564, 0.2762)

        transformations = [
            ToTensor(swap_color_axis=True),  # reshape to (C x H x W)
            Normalize(mean=mean, std=std),  # center by mean and divide by std
        ]

        if not validation:
            transformations.append(RandomHorizontalFlip(p=0.5))
            transformations.append(RandomCrop(size=32, padding=4, padding_mode="reflect"))
            transformations.append(RandomRotator(degrees=(0,15)))

        cifar_data.set_transformation(transforms.Compose(transformations))

        if not train:
            batch_size = self.batch_sizes["test"]
            dataloader = DataLoader(cifar_data, batch_size=batch_size, shuffle=True, num_workers=self.num_workers)
            return cifar_data, dataloader

        train_indices, validation_indices = self.get_validation_and_train_indices(cifar_data)
        indices = validation_indices if validation else train_indices
        subsample_cifar_data_set(sub_sample_indices=indices, cifar_data=cifar_data)
        batch_size = self.batch_sizes["validation"] if validation else self.batch_sizes["train"]
        return cifar_data, DataLoader(cifar_data, batch_size=batch_size, shuffle=True, num_workers=self.num_workers)

    def get_validation_and_train_indices(self, cifar_data: CifarDataSet):
        """
        Splits the cifar data into validation and train set and returns the indices of each set with respect to the
        original dataset
        :param cifar_data: and instance of CifarDataSet
        :return: train and validation indices
        """
        num_val_samples_per_class = 50
        num_train_samples_per_class = 450
        validation_set_size = 5000
        train_set_size = 45000

        validation_indices = torch.zeros(validation_set_size, dtype=torch.int32)
        train_indices = torch.zeros(train_set_size, dtype=torch.int32)
        current_val_samples = 0
        current_train_samples = 0
        for i in range(self.num_classes):
            class_indices = torch.argwhere(cifar_data.data["labels"][:, i] == 1).flatten()
            validation_indices[current_val_samples:(current_val_samples + num_val_samples_per_class)] += class_indices[:num_val_samples_per_class]
            train_indices[current_train_samples:(current_train_samples + num_train_samples_per_class)] += class_indices[num_val_samples_per_class:]
            current_val_samples += num_val_samples_per_class
            current_train_samples += num_train_samples_per_class

        return train_indices, validation_indices

    def _compute_avg_loss_on_trainloader(self, train_dataloader: DataLoader):
        """
        计算模型在当前任务训练数据上的平均损失（不更新梯度）。
        与 online_expr.py 中的 compute_avg_loss 功能一致。
        """
        self.net.eval()
        total_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for sample in train_dataloader:
                images = sample["image"].to(self.device)
                labels = sample["label"].to(self.device)
                logits = self.net(images)
                # self.loss 是 CrossEntropyLoss(reduction="mean")，需要乘以 batch size 来累加
                loss_val = self.loss(logits, labels)
                total_loss += loss_val.item() * images.size(0)
                total_samples += images.size(0)
        self.net.train()
        return total_loss / total_samples if total_samples > 0 else float('inf')

    @staticmethod
    def _compute_param_diff_norm(state_dict_before, state_dict_after):
        """
        计算两个参数字典之间的总 L2 范数差（欧式距离）。
        与 online_expr.py 中的 compute_param_diff_norm 功能一致。
        返回 total_norm (float)
        """
        total_norm_sq = 0.0
        for key in state_dict_before.keys():
            diff = state_dict_after[key] - state_dict_before[key]
            total_norm_sq += torch.sum(diff ** 2).item()
        return math.sqrt(total_norm_sq)

    def train(self, train_dataloader: DataLoader, test_dataloader: DataLoader, val_dataloader: DataLoader,
              test_data: CifarDataSet, training_data: CifarDataSet, val_data: CifarDataSet):

        self._save_model_parameters()

        # ========== 初始化可塑性指标 CSV 文件 ==========
        plasticity_file = os.path.join(self.results_dir, 'plasticity.csv')
        if not os.path.exists(plasticity_file):
            with open(plasticity_file, 'w', newline="") as f:
                writer = csv.writer(f)
                writer.writerow(['task_idx', 'loss_pre', 'loss_post', 'delta_norm_total', 'P_n'])

        # ========== 第一个 task 的预评估（第一个 epoch 之前） ==========
        self.plasticity_loss_pre = self._compute_avg_loss_on_trainloader(train_dataloader)
        self.plasticity_params_before = deepcopy(self.net.state_dict())
        self._print("[Task {0}] Pre-assessment loss: {1:.6f}".format(
            self.current_task, self.plasticity_loss_pre))

        for e in tqdm(range(self.current_epoch, self.num_epochs)):
            self._print("\tEpoch number: {0}".format(e + 1))
            self.set_lr()

            epoch_start_time = time.perf_counter()
            epoch_train_correct, epoch_train_total = 0, 0
            for step_number, sample in enumerate(train_dataloader):
                # sample observation and target
                image = sample["image"].to(self.device)
                label = sample["label"].to(self.device)

                # reset gradients
                for param in self.net.parameters(): param.grad = None   # apparently faster than optim.zero_grad()

                # learner handles forward, loss, backward, optimizer step, and algorithm-specific logic
                current_loss, predictions = self.learner.learn(image, label)

                # store summaries
                current_accuracy = torch.mean((predictions.argmax(axis=1) == label.argmax(axis=1)).to(torch.float32))
                self.running_loss += current_loss
                self.running_accuracy += current_accuracy.detach()
                epoch_train_correct += (predictions.argmax(axis=1) == label.argmax(axis=1)).sum().item()
                epoch_train_total += image.size(0)
                if (step_number + 1) % self.running_avg_window == 0:
                    self._print("\t\tStep Number: {0}".format(step_number + 1))
                    self._store_training_summaries()

            epoch_end_time = time.perf_counter()
            # store train accuracy for this epoch
            train_acc = epoch_train_correct / epoch_train_total if epoch_train_total > 0 else 0.0
            self.results_dict["train_accuracy_per_epoch"][e] = train_acc
            self._store_test_summaries(test_dataloader, val_dataloader, epoch_number=e,
                                       epoch_runtime=epoch_end_time - epoch_start_time)

            self.current_epoch += 1

            # ========== 第一个 epoch 结束后，计算可塑性指标 ==========
            # 条件：刚完成的 epoch 是该 task 的第一个 epoch
            if (self.current_epoch - 1) % self.num_epochs_per_task == 0:
                loss_post = self._compute_avg_loss_on_trainloader(train_dataloader)
                params_after = self.net.state_dict()
                delta_norm_total = self._compute_param_diff_norm(
                    self.plasticity_params_before, params_after)
                P_n = (self.plasticity_loss_pre - loss_post) / (delta_norm_total + 1e-8)

                with open(plasticity_file, 'a', newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        self.current_task,
                        "{0:.6f}".format(self.plasticity_loss_pre),
                        "{0:.6f}".format(loss_post),
                        "{0:.6f}".format(delta_norm_total),
                        "{0:.6f}".format(P_n)
                    ])

                self._print("[Task {0}] Plasticity P_n = {1:.6f} (ΔL={2:.4f}, Δθ={3:.4f})".format(
                    self.current_task, P_n,
                    self.plasticity_loss_pre - loss_post, delta_norm_total))

            # --- 训练集准确率达标则提前结束当前 task ---
            if self.current_epoch % self.num_epochs_per_task != 0:
                recent_acc = self.results_dict["train_accuracy_per_epoch"][self.current_epoch - 1]
                if recent_acc >= self.target_accuracy:
                    next_task_boundary = ((self.current_epoch - 1) // self.num_epochs_per_task + 1) * self.num_epochs_per_task
                    self._print("\n\t=== Target train accuracy {0:.2%} reached (got {1:.4f}) at epoch {2}. "
                                "Skipping to next task. ===".format(
                                    self.target_accuracy, recent_acc, self.current_epoch))
                    self.current_epoch = next_task_boundary

            # --- save experiment checkpoint every 50 epochs (for crash recovery) ---
            if self.current_epoch % 50 == 0:
                self.save_experiment_checkpoint()

            # --- task switching: when a task finishes, move to the next one ---
            if self.current_epoch % self.num_epochs_per_task == 0:
                next_task = self.current_epoch // self.num_epochs_per_task

                # ========== 刚完成的 task 的 NC 指标（无论是否还有下一个 task） ==========
                self._print("\n\t=== Task {0} finished (epoch {1}) ===".format(
                    self.current_task, self.current_epoch))
                # evaluate on all seen tasks before switching
                self._evaluate_all_seen_tasks(test_data)
                # compute NC metrics for the just-finished task (stored for later CSV export)
                nc1, nc2, nc3, nc3_max, nc3_min, nc4, isotropy, equinormity = self.compute_nc_metrics(test_data)
                self.results_dict["nc1_per_task"][self.current_task] = nc1
                self.results_dict["nc2_per_task"][self.current_task] = nc2
                self.results_dict["nc3_per_task"][self.current_task] = nc3
                self.results_dict["nc3_max_per_task"][self.current_task] = nc3_max
                self.results_dict["nc3_min_per_task"][self.current_task] = nc3_min
                self.results_dict["nc4_per_task"][self.current_task] = nc4
                self.results_dict["isotropy_per_task"][self.current_task] = isotropy
                self.results_dict["equinormity_per_task"][self.current_task] = equinormity
                # save NC3 visualization data for t-SNE plotting
                self._save_nc3_data(test_data)
                # record best accuracy for the just-finished task
                self._print("\tBest accuracy in task {0}: {1:.4f}".format(
                    self.current_task, self.best_accuracy))
                if self.early_stopping:
                    self.net.load_state_dict(self.best_accuracy_model_parameters)
                self.best_accuracy = torch.zeros_like(self.best_accuracy)
                self.best_accuracy_model_parameters = {}
                self._save_model_parameters()

                # algorithm-specific post-task hook (runs on the JUST-FINISHED task's data)
                self.learner.after_task(train_dataloader)

                if next_task < self.num_tasks:
                    self._print("\n\t=== Switching to Task {0} ===".format(next_task))

                    # switch data to next task
                    self.current_task = next_task
                    self._restore_from_base(training_data, self.base_data["train"])
                    self._restore_from_base(val_data, self.base_data["val"])
                    self._restore_from_base(test_data, self.base_data["test"])
                    self._remap_labels_to_task(training_data, self.current_task)
                    self._remap_labels_to_task(val_data, self.current_task)
                    self._remap_labels_to_task(test_data, self.current_task)
                    # recreate dataloaders (workers need fresh dataset copies)
                    train_dataloader = self._create_dataloader(training_data, "train")
                    val_dataloader = self._create_dataloader(val_data, "validation")
                    test_dataloader = self._create_dataloader(test_data, "test")
                    self._print("\tNew task data loaded: {0} training samples\n".format(
                        len(training_data.data["data"])))

                    # ========== 新 task 的预评估（第一个 epoch 之前） ==========
                    self.plasticity_loss_pre = self._compute_avg_loss_on_trainloader(train_dataloader)
                    self.plasticity_params_before = deepcopy(self.net.state_dict())
                    self._print("[Task {0}] Pre-assessment loss: {1:.6f}".format(
                        self.current_task, self.plasticity_loss_pre))


    def set_lr(self):
        """ Changes the learning rate of the optimizer according to the current epoch within the task """
        current_stepsize = None
        epoch_in_task = self.current_epoch % self.num_epochs_per_task
        if epoch_in_task == 0:
            current_stepsize = self.stepsize
        elif epoch_in_task == 60:
            current_stepsize = round(self.stepsize * 0.2, 5)
        elif epoch_in_task == 120:
            current_stepsize = round(self.stepsize * (0.2 ** 2), 5)
        elif epoch_in_task == 160:
            current_stepsize = round(self.stepsize * (0.2 ** 3), 5)

        if current_stepsize is not None:
            for g in self.optim.param_groups:
                g['lr'] = current_stepsize
            self._print("\tCurrent stepsize: {0:.5f}".format(current_stepsize))

    def _evaluate_all_seen_tasks(self, test_data: CifarDataSet):
        """
        Evaluate the current model on all tasks seen so far, using the pre-stored
        per-task test data snapshots. Records accuracies in self.task_accuracies.
        This allows measuring plasticity (latest task) and forgetting (earlier tasks).

        :param test_data: current test CifarDataSet (will be temporarily modified and restored)
        """
        # save current test data state
        current_test_data = test_data.data["data"]
        current_test_labels = test_data.data["labels"]
        current_int_labels = list(test_data.integer_labels)

        self.net.eval()
        for task_id in range(self.current_task + 1):
            task_snapshot = self.per_task_test_data[task_id]

            # temporarily inject task_id's test data into the CifarDataSet
            test_data.data["data"] = task_snapshot["data"]
            test_data.data["labels"] = task_snapshot["labels"]
            test_data.integer_labels = [np.argmax(l) for l in task_snapshot["labels"]]
            test_data.current_data = test_data.partition_data()

            # evaluate on this task
            task_loader = self._create_dataloader(test_data, "test")
            avg_loss, avg_acc = self.evaluate_network(task_loader)

            self.task_accuracies[self.current_task, task_id] = avg_acc
            self._print("\t\tTask {0} accuracy: {1:.4f}".format(task_id, avg_acc))

        self.net.train()

        # restore current task's test data
        test_data.data["data"] = current_test_data
        test_data.data["labels"] = current_test_labels
        test_data.integer_labels = current_int_labels
        test_data.current_data = test_data.partition_data()

    def _save_model_parameters(self):
        """ Stores the parameters of the model, so it can be evaluated after the experiment is over """

        model_parameters_dir_path = os.path.join(self.results_dir, "model_parameters")
        os.makedirs(model_parameters_dir_path, exist_ok=True)

        file_name = "index-{0}_epoch-{1}.pt".format(self.run_index, self.current_epoch)
        file_path = os.path.join(model_parameters_dir_path, file_name)

        store_object_with_several_attempts(self.net.state_dict(), file_path, storing_format="torch", num_attempts=10)

    def _save_task_accuracies_csv(self):
        """
        Save the task_accuracies matrix as a CSV file for easy reading.
        Rows = after training on task t, Columns = accuracy on task k.
        """
        import csv

        csv_dir = os.path.join(self.results_dir, "task_accuracies")
        os.makedirs(csv_dir, exist_ok=True)
        csv_path = os.path.join(csv_dir, "index-{0}.csv".format(self.run_index))

        acc = self.task_accuracies.cpu().numpy()
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["After Task \\ On Task"] + ["Task {0}".format(i) for i in range(acc.shape[1])])
            for t in range(acc.shape[0]):
                writer.writerow(["Task {0}".format(t)] + ["{0:.4f}".format(v) for v in acc[t]])

        self._print("\tTask accuracies CSV saved to: {0}".format(csv_path))


def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', action="store", type=str,
                        default='',
                        help="Path to the config file. Defaults to <script_dir>/cfg/task_continual_learning.json")
    parser.add_argument("--experiment-index", action="store", type=int, default=0,
                        help="Index for the run; this will determine the random seed and the name of the results.")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Whether to print extra information about the experiment as it's running.")
    args = parser.parse_args()

    file_path = os.path.dirname(os.path.abspath(__file__))
    if args.config == "":
        args.config = os.path.join(file_path, "cfg", "task_continual_learning.json")

    with open(args.config, 'r', encoding='utf-8') as config_file:
        experiment_parameters = json.load(config_file)
    if "data_path" not in experiment_parameters.keys() or experiment_parameters["data_path"] == "":
        experiment_parameters["data_path"] = os.path.join(file_path, "data")
    if "results_dir" not in experiment_parameters.keys() or experiment_parameters["results_dir"] == "":
        experiment_parameters["results_dir"] = os.path.join(file_path, "results")
    if "experiment_name" not in experiment_parameters.keys() or experiment_parameters["experiment_name"] == "":
        experiment_parameters["experiment_name"] = os.path.splitext(os.path.basename(args.config))[0]

    initial_time = time.perf_counter()
    exp = IncrementalCIFARExperiment(experiment_parameters,
                                     results_dir=os.path.join(experiment_parameters["results_dir"], experiment_parameters["experiment_name"]),
                                     run_index=args.experiment_index,
                                     verbose=args.verbose)
    exp.run()
    exp.store_results()
    final_time = time.perf_counter()
    print("The running time in minutes is: {0:.2f}".format((final_time - initial_time) / 60))


if __name__ == "__main__":
    main()
