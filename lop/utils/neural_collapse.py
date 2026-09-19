"""
Neural collapse metrics

This module contains functions computing the metrics (NC1-NC4) introduced in Papyan et al. (2020).
The code was adapted from Zhu et al. (2021): https://github.com/tding1/Neural-Collapse
"""
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Tuple, Dict, Optional


# ------------------------------
# Helpers
# ------------------------------

def _rebatch_loader(loader: DataLoader, max_bs: int = 128) -> DataLoader:
    """
    Rebuild a dataloader with a (possibly) smaller batch size to reduce GPU peak memory.
    Keeps dataset/shuffle/drop_last/pin_memory settings if available.
    """
    bs = getattr(loader, "batch_size", None)
    if bs is None or bs > max_bs:
        return DataLoader(
            loader.dataset,
            batch_size=max_bs,
            shuffle=getattr(loader, "shuffle", False),
            drop_last=getattr(loader, "drop_last", False),
            num_workers=getattr(loader, "num_workers", 0),
            pin_memory=getattr(loader, "pin_memory", False),
        )
    return loader


@torch.no_grad()


def _extract_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    model.eval()
    device = next(model.parameters()).device
    h = x.to(device)  # 直接用原始输入

    # ResNet / 有 embed 方法的模型：直接调用 embed
    if hasattr(model, 'embed'):
        return model.embed(h)

    # MLP / 有 layers 属性的模型：按层级顺序前向
    if hasattr(model, 'layers'):
        h = model.layers[0](h)
        if len(model.layers) > 1 and isinstance(model.layers[1], nn.ReLU):
            h = model.layers[1](h)
        h = model.layers[2](h)
        if len(model.layers) > 3 and isinstance(model.layers[3], nn.ReLU):
            h = model.layers[3](h)
        h = model.layers[4](h)
        if len(model.layers) > 5 and isinstance(model.layers[5], nn.ReLU):
            h = model.layers[5](h)
        return h

    # 兜底：尝试直接用模型前向（不经过分类头）
    raise AttributeError(
        "Model '{0}' has neither 'embed' nor 'layers' attribute. "
        "Please ensure the model exposes feature extraction via embed() or layers."
        .format(type(model).__name__))


# ------------------------------
# 新增：分类器权重提取函数（适配DeepFFNN）
# ------------------------------
@torch.no_grad()
def _get_classifier_weights(model: nn.Module) -> torch.Tensor:
    """
    适配DeepFFNN模型：提取输出层（layers.6）的权重
    确保权重维度为 [num_classes, feature_dim]
    """
    # ResNet 风格：直接取 fc 分类头（显式分支，避免走异常回退路径）
    if hasattr(model, 'fc') and isinstance(getattr(model, 'fc'), nn.Linear):
        return model.fc.weight.data.clone()

    # 优先直接取输出层权重（从你的CSV保存代码可知输出层是layers.6）
    try:
        output_layer = model.layers[6]
        W = output_layer.weight.data.clone()
    except (IndexError, AttributeError):
        # 备用方案：找最后一个Linear层
        linear_layers = [m for _, m in model.named_modules() if isinstance(m, nn.Linear)]
        if not linear_layers:
            raise ValueError("Model has no Linear layer for classification!")
        W = linear_layers[-1].weight.data.clone()
    
    # 验证权重维度（至少2类）
    assert W.shape[0] >= 2, f"输出层权重维度错误，期望[num_classes, D]，实际{W.shape}"
    return W


# ------------------------------
# Means (single pass, no repeated large allocs)
# ------------------------------

@torch.no_grad()
def _get_feature_means(
    model: nn.Module,
    data_loader: DataLoader,
    num_classes: int,
    use_cache: bool = False
) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
    """
    Return global mean and dict of per-class means on the model's device.

    Memory-friendly implementation:
    - Do NOT allocate zeros per missing class in each batch.
    - Accumulate sums + counts, then divide once at the end.
    """
    if use_cache:
        if hasattr(_get_feature_means, "cached_mu_G") and hasattr(_get_feature_means, "cached_mu_c_dict"):
            return _get_feature_means.cached_mu_G, _get_feature_means.cached_mu_c_dict

    device = next(model.parameters()).device
    model.eval()

    # First batch to infer feature dimension D
    # 注意：必须复用同一个迭代器，否则下面的 for 循环会从头再迭代一遍，
    # 导致第一个 batch 被重复计入均值（double-count bug）。
    loader_iter = iter(data_loader)
    first_inputs, first_targets = next(loader_iter)
    first_inputs = first_inputs.to(device, non_blocking=True)
    first_feats = _extract_features(model, first_inputs)
    D = first_feats.shape[1]

    mu_sum = torch.zeros(D, device=device, dtype=first_feats.dtype)
    class_sum = torch.zeros(num_classes, D, device=device, dtype=first_feats.dtype)
    class_cnt = torch.zeros(num_classes, device=device, dtype=torch.long)

    # process the first batch
    mu_sum += first_feats.sum(dim=0)
    t = first_targets.to(device, non_blocking=True).long()
    for c in range(num_classes):
        mask = (t == c)
        if mask.any():
            class_sum[c] += first_feats[mask].sum(dim=0)
            class_cnt[c] += mask.sum()

    # remaining batches（复用上面的迭代器，避免首批被重复计入）
    for inputs, targets in loader_iter:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).long()
        feats = _extract_features(model, inputs)

        mu_sum += feats.sum(dim=0)
        for c in range(num_classes):
            m = (targets == c)
            if m.any():
                class_sum[c] += feats[m].sum(dim=0)
                class_cnt[c] += m.sum()

        del feats, inputs, targets
        if device.type == "cuda":
            torch.cuda.empty_cache()

    N = len(data_loader.dataset)
    mu_G = mu_sum / max(N, 1)

    mu_c_dict: Dict[int, torch.Tensor] = {}
    for c in range(num_classes):
        count = class_cnt[c].item()
        if count > 0:
            mu_c_dict[c] = class_sum[c] / count
        else:
            mu_c_dict[c] = torch.zeros(D, device=device, dtype=class_sum.dtype)

    # cache
    _get_feature_means.cached_mu_G = mu_G
    _get_feature_means.cached_mu_c_dict = mu_c_dict

    return mu_G, mu_c_dict


def clear_feature_means_cache():
    """清除 _get_feature_means 的缓存（切换模型/task 时必须调用）"""
    if hasattr(_get_feature_means, "cached_mu_G"):
        del _get_feature_means.cached_mu_G
    if hasattr(_get_feature_means, "cached_mu_c_dict"):
        del _get_feature_means.cached_mu_c_dict


# ------------------------------
# NC1: Within / Between covariance (two-pass, batchwise)
# ------------------------------

@torch.no_grad()

def NC1(
    model: nn.Module,
    num_classes: int,
    inputs: Optional[torch.Tensor] = None,
    targets: Optional[torch.Tensor] = None,
    data_loader: Optional[DataLoader] = None,
    use_cache: bool = False
) -> float:
    """
    Cross-example within-class variability: Tr(Sigma_W Sigma_B^+)/(K)
    """
    if data_loader is None:
        assert inputs is not None and targets is not None, "no data provided"
        data_loader = DataLoader(list(zip(inputs, targets)), batch_size=len(targets))

    device = next(model.parameters()).device
    model.eval()

    loader = _rebatch_loader(data_loader, max_bs=128)

    mu_G, mu_c_dict = _get_feature_means(model=model, data_loader=loader, num_classes=num_classes, use_cache=use_cache)
    D = mu_G.shape[0]

    # ---- Sigma_W ----
    Sigma_W = torch.zeros(D, D, device=device, dtype=mu_G.dtype)
    N = len(loader.dataset)

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long()
        feats = _extract_features(model, x)

        for c in range(num_classes):
            m = (y == c)
            if m.any():
                diff = feats[m] - mu_c_dict[c]
                Sigma_W += diff.transpose(0, 1) @ diff
        del feats, x, y, diff, m
        if device.type == "cuda":
            torch.cuda.empty_cache()

    Sigma_W = Sigma_W / max(N, 1)

    # ---- Sigma_B ----
    Sigma_B = torch.zeros(D, D, device=device, dtype=mu_G.dtype)
    for c in range(num_classes):
        d = (mu_c_dict[c] - mu_G).unsqueeze(1)
        Sigma_B += d @ d.transpose(0, 1)
    Sigma_B = Sigma_B / max(num_classes, 1)

    # ---- 稳定伪逆：用 eigh 替代 SVD（对称矩阵特征分解更稳定）----
    # Sigma_B 秩 <= K-1 = 9，远小于 D=512，SVD 极易不收敛
    Sigma_B_sym = (Sigma_B + Sigma_B.T) / 2  # 确保数值对称
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(Sigma_B_sym)
        # 截断小特征值：保留大于最大特征值 1e-6 倍的
        max_eig = eigenvalues.abs().max()
        threshold = max_eig * 1e-6 if max_eig > 0 else torch.tensor(1e-10, device=eigenvalues.device)
        inv_eigenvalues = torch.where(
            eigenvalues > threshold,
            1.0 / eigenvalues.clamp(min=1e-10),
            torch.zeros_like(eigenvalues)
        )
        pinv_SB = eigenvectors @ torch.diag(inv_eigenvalues) @ eigenvectors.T
    except Exception:
        # eigh 也失败时（极端情况），用强正则化兜底
        reg = torch.trace(Sigma_B).abs() / Sigma_B.shape[0] * 1e-2 + 1e-6
        pinv_SB = torch.linalg.pinv(Sigma_B + reg * torch.eye(Sigma_B.shape[0], device=Sigma_B.device, dtype=Sigma_B.dtype))
    val = torch.trace(Sigma_W @ pinv_SB) / num_classes
    return val.item()


# ------------------------------
# NC2: Simplex ETF convergence (class means-based, align with paper)
# ------------------------------

@torch.no_grad()
def NC2(
    model: nn.Module,
    num_classes: int,
    inputs: Optional[torch.Tensor] = None,
    targets: Optional[torch.Tensor] = None,
    data_loader: Optional[DataLoader] = None,
    use_cache: bool = False
) -> Tuple[float, float, float]:
    """
    Align with Papyan et al. (2020) NC2 definition:
    Measure convergence of centered class means to simplex ETF via two core indicators:
    1. Coefficient of variation of class mean norms (equinorm property)
    2. Standard deviation of inter-class mean cosines (equiangular property)
    Return combined score (smaller = better NC2 convergence)
    """
    if data_loader is None:
        assert inputs is not None and targets is not None, "no data provided"
        data_loader = DataLoader(list(zip(inputs, targets)), batch_size=len(targets))

    device = next(model.parameters()).device
    loader = _rebatch_loader(data_loader, max_bs=128)

    # Get centered class means (core object of paper's NC2)
    mu_G, mu_c_dict = _get_feature_means(model=model, data_loader=loader, num_classes=num_classes, use_cache=use_cache)
    mu_centered = torch.stack([mu_c_dict[c] - mu_G for c in range(num_classes)], dim=0).to(device)  # [K, D]

    # 1. Equinorm indicator: Coefficient of variation (std / avg) of class mean norms
    norms = torch.norm(mu_centered, p=2, dim=1)  # [K]
    avg_norm = torch.mean(norms)
    std_norm = torch.std(norms)
    cv_norm = std_norm / (avg_norm + 1e-8)  # Avoid division by zero

    # 2. Equiangular indicator: Std of inter-class cosine similarities
    mu_normalized = mu_centered / (norms.unsqueeze(1) + 1e-8)  # Normalize to unit norm
    cos_matrix = mu_normalized @ mu_normalized.T  # [K, K]
    # Extract off-diagonal elements (c != c')
    off_diag_mask = torch.triu(torch.ones_like(cos_matrix, dtype=bool), diagonal=1)
    cos_vals = cos_matrix[off_diag_mask]
    std_cos = torch.std(cos_vals) if len(cos_vals) > 0 else torch.tensor(0.0, device=device)

    # Combine indicators (consistent with paper's dual conditions, smaller = better)
    nc2_score = torch.sqrt(cv_norm ** 2 + std_cos ** 2).item()
    equinorm = cv_norm.item()  # 等范数性：变异系数
    equiangular = std_cos.item()  # 等角性：余弦标准差
    print(f"[nc代码记录nc2  ]  {equinorm:.4f}， { equiangular:.4f}")
    return nc2_score, equinorm, equiangular


# ------------------------------
# NC3: Duality (修正为原版论文逻辑)
# ------------------------------
@torch.no_grad()
def NC3(
    model: nn.Module,
    num_classes: int,
    inputs: Optional[torch.Tensor] = None,
    targets: Optional[torch.Tensor] = None,
    data_loader: Optional[DataLoader] = None,
    use_cache: bool = False
) -> Tuple[float, float, float]:
    if data_loader is None:
        assert inputs is not None and targets is not None, "no data provided"
        data_loader = DataLoader(list(zip(inputs, targets)), batch_size=len(targets))

    device = next(model.parameters()).device
    loader = _rebatch_loader(data_loader, max_bs=128)

    # 1. 获取均值（保持你原逻辑）
    mu_G, mu_c_dict = _get_feature_means(
        model=model,
        data_loader=loader,
        num_classes=num_classes,
        use_cache=use_cache
    )

    # 2. 分类器权重 W: [K, D]
    W = _get_classifier_weights(model).to(device)

    # 3. 构建中心化类均值（保持你原逻辑）
    M = torch.stack([mu_c_dict[i] - mu_G for i in range(num_classes)], dim=0).to(device)  # [K, D]

    
    eps = 1e-8

    # 4. 逐类归一化
    W_norm = W / (torch.norm(W, dim=1, keepdim=True) + eps)
    M_norm = M / (torch.norm(M, dim=1, keepdim=True) + eps)

    # 5. 逐类 cosine
    cos_sim = torch.sum(W_norm * M_norm, dim=1)  # [K]

    # 6. NC3_k = 1 - cos
    nc3_per_class = 1.0 - cos_sim

    # # 7. 平均
    # nc3_score = nc3_per_class.mean().item()

    # return nc3_score

    # ====================== 核心改动 ======================
    nc3_mean = nc3_per_class.mean().item()
    nc3_max  = nc3_per_class.max().item()
    nc3_min  = nc3_per_class.min().item()

    print(f"[nc3代码记录  ]  {nc3_mean:.4f}，{nc3_max:.4f}，{nc3_min:.4f}")
    # 返回三个值！
    return nc3_mean, nc3_max, nc3_min


# ------------------------------
# NC4: Agreement with nearest class center (vectorized on GPU)
# ------------------------------

@torch.no_grad()
def NC4(
    model: nn.Module,
    num_classes: int,
    inputs: Optional[torch.Tensor] = None,
    targets: Optional[torch.Tensor] = None,
    data_loader: Optional[DataLoader] = None,
    use_cache: bool = False
) -> float:
    if data_loader is None:
        assert inputs is not None and targets is not None, "no data provided"
        data_loader = DataLoader(list(zip(inputs, targets)), batch_size=len(targets))

    device = next(model.parameters()).device
    model.eval()

    loader = _rebatch_loader(data_loader, max_bs=128)
    _, mu_c_dict = _get_feature_means(model=model, data_loader=loader, num_classes=num_classes, use_cache=use_cache)

    centers = torch.stack([mu_c_dict[i] for i in range(num_classes)], dim=0).to(device)  # [K, D]
    centers_sq = (centers ** 2).sum(dim=1, keepdim=True)  # [K,1]

    agree = 0
    total = 0
    for x, _ in loader:
        x = x.to(device, non_blocking=True)

        feats = _extract_features(model, x)  # [B, D]
        logits = model(x)                    # [B, K]

        # nearest center (squared Euclidean)
        # dist^2(x, c) = ||x||^2 - 2 x @ c^T + ||c||^2
        x_sq = (feats ** 2).sum(dim=1, keepdim=True)       # [B,1]
        xc = feats @ centers.T                              # [B,K]
        d2 = x_sq - 2 * xc + centers_sq.T                   # [B,K]
        nn_center = torch.argmin(d2, dim=1)                 # [B]

        pred = torch.argmax(logits, dim=1)                  # [B]
        agree += (nn_center == pred).sum().item()
        total += x.shape[0]

        del feats, logits, x, x_sq, xc, d2, nn_center, pred
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return float(agree / max(total, 1))


# ------------------------------
# NC wrapper (修正缓存逻辑+强制eval模式)
# ------------------------------

@torch.no_grad()
def NC(
    model: nn.Module,
    num_classes: int,
    inputs: Optional[torch.Tensor] = None,
    targets: Optional[torch.Tensor] = None,
    data_loader: Optional[DataLoader] = None,
    use_cache: bool = False
) -> Tuple[float, float, float, float, float, float, float, float]:
    model.eval()
    
    if data_loader is None:
        assert inputs is not None and targets is not None
        data_loader = DataLoader(list(zip(inputs, targets)), batch_size=len(targets))

    loader = _rebatch_loader(data_loader, max_bs=128)

    # ====================== 缓存优化 ======================
    # 先清除旧模型/旧数据的特征均值缓存，然后在本次 NC 评估内部
    # 让 NC1-NC4 共享同一轮特征均值计算（use_cache=True）。
    # 否则每个指标各自把全数据集过一遍特征提取（ResNet 上开销 x4）。
    clear_feature_means_cache()
    inner_cache = True

    nc1 = NC1(model=model, data_loader=loader, num_classes=num_classes, use_cache=inner_cache)
    nc2, nc_equinorm, nc_equiangular = NC2(model=model, data_loader=loader, num_classes=num_classes, use_cache=inner_cache)
    nc3, nc3_max, nc3_min = NC3(model=model, data_loader=loader, num_classes=num_classes, use_cache=inner_cache)
    nc4 = NC4(model=model, data_loader=loader, num_classes=num_classes, use_cache=inner_cache)
    # 本次评估结束后清除缓存，防止下一次 NC 调用（模型已更新）复用旧均值
    clear_feature_means_cache()
    # ====================== 【唯一正确的清理代码】 ======================
    import gc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    # ====================================================================

    print(f"[nc代码记录2  ]  {nc_equinorm:.4f}， { nc_equiangular:.4f},{nc1:.4f}，{nc2:.4f}，{nc3:.4f}，{nc3_max:.4f}，{nc3_min:.4f}")

    return nc1, nc2, nc3, nc3_max, nc3_min, nc4, nc_equiangular, nc_equinorm