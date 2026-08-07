"""
Neural collapse metrics

This module contains functions computing the metrics (NC1-NC4) introduced in Papyan et al. (2020).
The code was adapted from Zhu et al. (2021): https://github.com/tding1/Neural-Collapse
"""
import numpy as np
import torch
import pdb
from torch import nn
from torch.utils.data import DataLoader
from typing import Tuple, Dict
import gc  # 新增：导入垃圾回收模块（仅新增这一行）


def _get_feature_means(model: nn.Module,
                       data_loader: DataLoader,
                       num_classes: int,
                       use_cache: bool = False) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
    # returns global mean and dict of class means

    if use_cache:
        # check if cached values are available
        if hasattr(_get_feature_means, 'cached_mu_G') and hasattr(_get_feature_means, 'cached_mu_c_dict'):
            return _get_feature_means.cached_mu_G, _get_feature_means.cached_mu_c_dict

    mu_G = 0
    mu_c_dict = dict()
    samples_per_class = np.zeros(num_classes)
    device = next(model.parameters()).device
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        features = model.embed(inputs)
        samples_per_class += [sum(targets == i).cpu().item() for i in range(num_classes)]
        
        # global mean
        mu_G = mu_G + torch.sum(features, dim=0)

        # class means
        for b in range(len(targets)):
            y = targets[b].item()
            if y not in mu_c_dict:
                mu_c_dict[y] = features[b, :]
            else:
                mu_c_dict[y] = mu_c_dict[y] + features[b, :]

        # in case some class was not present in batch assign zero vector
        for y in range(num_classes):
            if y not in mu_c_dict:
                mu_c_dict[y] = torch.zeros_like(mu_G)

    mu_G = mu_G / len(data_loader.dataset)
    for i in range(num_classes):
        if samples_per_class[i] > 0:
            mu_c_dict[i] = mu_c_dict[i] / samples_per_class[i]

    # cache the computed values
    _get_feature_means.cached_mu_G = mu_G
    _get_feature_means.cached_mu_c_dict = mu_c_dict

    # 新增：清理临时张量和GPU缓存（最小幅度修改）
    del features, inputs, targets  # 删除当前批次的临时大张量
    torch.cuda.empty_cache()  # 清理GPU未使用的缓存
    gc.collect()  # 触发Python垃圾回收

    return mu_G, mu_c_dict


def _get_classifier_weights(model: torch.nn.Module) -> torch.Tensor:
    # returns weights of last linear layer
##w    linear_layers = [layer for layer in model.children() if isinstance(layer, nn.Linear)]
#    for name, module in model.named_modules():
#        print(f"Name: {name}, Type: {type(module).__name__}")
#
#    for name, module in model.named_modules():
#        print(f"Layer name: {name}")
#        print(f"Module type: {type(module).__name__}")
#
#        # if Linear, print the shapes of weights and bias
#        if isinstance(module, nn.Linear):
#            print(f"  Weight shape: {module.weight.shape}")
#            print(f"  Bias shape: {module.bias.shape}")
#
#        # if ModuleList, print its children moduls
#        if isinstance(module, nn.ModuleList):
#            for i, submodule in enumerate(module):
#                print(f"  Submodule {i}: {type(submodule).__name__}")
#                if isinstance(submodule, nn.Linear):
#                    print(f"    Weight shape: {submodule.weight.shape}")
#
#        print("-" * 50)

    linear_layers = [module for name, module in model.named_modules() if isinstance(module, nn.Linear)]

    return linear_layers[-1].state_dict()['weight']


def NC1(model: nn.Module,
        num_classes: int,
        inputs: torch.Tensor = None,
        targets: torch.Tensor = None,
        data_loader: DataLoader = None,
        use_cache: bool = False) -> float:
    """
    Compute NC1 (cross-example within-class variability).

    :param model: model with 'embed' function
    :param num_classes: number of distinct classes
    :param inputs: batch of data
    :param targets: ground truth labels
    :param data_loader: data loader (if provided, 'inputs' and 'targets' are ignored)
    :return: NC1
    """
    if data_loader is None:
        assert inputs is not None and targets is not None, "no data provided"
        data_loader = DataLoader(list(zip(inputs, targets)), batch_size=len(targets))

    device = next(model.parameters()).device

    # global mean and dict of class means
    mu_G, mu_c_dict = _get_feature_means(model=model, data_loader=data_loader, num_classes=num_classes,
                                         use_cache=use_cache)

    # within-class covariance
    Sigma_W = 0
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        features = model.embed(inputs)

        for b in range(len(targets)):
            y = targets[b].item()
            Sigma_W = Sigma_W + (features[b, :] - mu_c_dict[y]).unsqueeze(1) @ (
                        features[b, :] - mu_c_dict[y]).unsqueeze(0)

    Sigma_W = Sigma_W / len(data_loader.dataset)

    # between-class covariance
    Sigma_B = 0
    for i in range(num_classes):
        Sigma_B = Sigma_B + (mu_c_dict[i] - mu_G).unsqueeze(1) @ (mu_c_dict[i] - mu_G).unsqueeze(0)

    Sigma_B = Sigma_B / num_classes

    nc1_result = torch.trace(Sigma_W @ torch.linalg.pinv(Sigma_B)) / num_classes

    # 新增：清理临时张量和GPU缓存（最小幅度修改）
    del Sigma_W, Sigma_B, features, inputs, targets  # 删除大张量
    torch.cuda.empty_cache()
    gc.collect()

    return nc1_result


def NC2(model: nn.Module) -> float:
    """
    Compute NC2 (distance of last-layer classifier to a Simplex ETF).

    :param model: model with fully connected last layer
    :return: NC2
    """

    device = next(model.parameters()).device

    W = _get_classifier_weights(model)

    # compute ETF metric (see Zhu et al. (2021))
    K = W.shape[0]
    WWT = torch.mm(W, W.T)
    WWT /= torch.norm(WWT, p='fro')

    sub = (torch.eye(K) - 1 / K * torch.ones((K, K))).to(device) / pow(K - 1, 0.5)

    nc2_result = torch.norm(WWT - sub, p='fro').item()

    # 新增：清理临时张量和GPU缓存（最小幅度修改）
    del W, WWT, sub
    torch.cuda.empty_cache()
    gc.collect()

    return nc2_result


def NC3(model: nn.Module,
        num_classes: int,
        inputs: torch.Tensor = None,
        targets: torch.Tensor = None,
        data_loader: DataLoader = None,
        use_cache: bool = False) -> float:
    """
    Compute NC3 (distance of learned features to dual classifier)

    :param model: model with 'embed' function
    :param num_classes: number of distinct classes
    :param inputs: batch of data
    :param targets: ground truth labels
    :param data_loader: data loader (if provided, 'inputs' and 'targets' are ignored)
    :return: NC3
    """
    if data_loader is None:
        assert inputs is not None and targets is not None, "no data provided"
        data_loader = DataLoader(list(zip(inputs, targets)), batch_size=len(targets))

    device = next(model.parameters()).device

    mu_G, mu_c_dict = _get_feature_means(model=model, data_loader=data_loader, num_classes=num_classes,
                                         use_cache=use_cache)
    W = _get_classifier_weights(model)

    K = num_classes
    H = torch.empty(mu_c_dict[0].shape[0], K)
    for i in range(K):
        H[:, i] = mu_c_dict[i] - mu_G

    WH = torch.mm(W, H.to(device))
    WH /= torch.norm(WH, p='fro')
    sub = 1 / pow(K - 1, 0.5) * (torch.eye(K) - 1 / K * torch.ones((K, K))).to(device)

    res = torch.norm(WH - sub, p='fro')
    nc3_result = res.detach().cpu().numpy().item()

    # 新增：清理临时张量和GPU缓存（最小幅度修改）
    del H, WH, sub, res, W, mu_G
    torch.cuda.empty_cache()
    gc.collect()

    return nc3_result


def NC4(model: nn.Module,
        num_classes: int,
        inputs: torch.Tensor = None,
        targets: torch.Tensor = None,
        data_loader: DataLoader = None,
        use_cache: bool = False) -> float:
    """
    Compute NC4 (proportion of classifications agreeing with nearest center classifier)

    :param model: model with 'embed' function
    :param num_classes: number of distinct classes
    :param inputs: batch of data
    :param targets: ground truth labels
    :param data_loader: data loader (if provided, 'inputs' and 'targets' are ignored)
    :return: NC4
    """
    if data_loader is None:
        assert inputs is not None and targets is not None, "no data provided"
        data_loader = DataLoader(list(zip(inputs, targets)), batch_size=len(targets))

    device = next(model.parameters()).device

    _, mu_c_dict = _get_feature_means(model=model, data_loader=data_loader, num_classes=num_classes,
                                      use_cache=use_cache)

    class_centers = [value.cpu().detach() for (_, value) in sorted(mu_c_dict.items())]

    nearest_centers, preds = [], []

    for inputs, _ in data_loader:
        inputs = inputs.to(device)
        features = model.embed(inputs).detach()
        nearest_centers.extend(
            [np.argmin(list(map(lambda x: torch.norm(x - ft.cpu()), class_centers))) for ft in features])

        preds.extend(torch.argmax(model(inputs).detach(), dim=1).cpu().numpy())
    nearest_centers, preds = np.array(nearest_centers), np.array(preds)

    nc4_result = float((nearest_centers == preds).mean())

    # 新增：清理临时张量和GPU缓存（最小幅度修改）
    del features, inputs, class_centers, nearest_centers, preds
    torch.cuda.empty_cache()
    gc.collect()

    return nc4_result

def NC(model: nn.Module,
       num_classes: int,
       inputs: torch.Tensor = None,
       targets: torch.Tensor = None,
       data_loader: DataLoader = None) -> Tuple[float, float, float, float]:
    """
    Compute all NC metrics.

    :param model: model with 'embed' function
    :param num_classes: number of distinct classes
    :param inputs: batch of data
    :param targets: ground truth labels
    :param data_loader: data loader (if provided, 'inputs' and 'targets' are ignored)
    :return: NC1, NC2, NC3, NC4
    """
    if data_loader is None:
        assert inputs is not None and targets is not None, "no data provided"
        data_loader = DataLoader(list(zip(inputs, targets)), batch_size=len(targets))

    nc1 = NC1(model=model, data_loader=data_loader, num_classes=num_classes)
    nc2 = NC2(model=model)
    nc3 = NC3(model=model, data_loader=data_loader, num_classes=num_classes, use_cache=True)
    nc4 = NC4(model=model, data_loader=data_loader, num_classes=num_classes, use_cache=True)

    # 新增：最终清理（最小幅度修改）
    torch.cuda.empty_cache()
    gc.collect()

    return nc1.cpu().item(), nc2, nc3, nc4