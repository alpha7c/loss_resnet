# cifar10_data.py
import torch
import torchvision
import torchvision.transforms as transforms

def load_cifar10(device='cpu', flatten=True, use_grayscale=False, normalize=False):
    """
    加载 CIFAR-10 数据集，返回与原有 MNIST 加载格式完全兼容的张量。

    Args:
        device: 目标设备 ('cpu' 或 'cuda')
        flatten: 是否将图像展平为一维向量
        use_grayscale: 是否转换为灰度图（默认 False，保留 RGB 三通道）
        normalize: 是否用 CIFAR-10 均值/方差归一化（ResNet 训练建议开启）。
                   注意：归一化在展平/像素排列之前完成，因此排列后的任务
                   数据依然保持归一化分布。

    Returns:
        x_train: torch.Tensor, shape [50000, D] 或 [50000, C, H, W]
        y_train: torch.Tensor, shape [50000]
        x_test:  torch.Tensor, shape [10000, D] 或 [10000, C, H, W]
        y_test:  torch.Tensor, shape [10000]
    """
    if use_grayscale:
        # 灰度图：32x32 = 1024 维
        transform_list = [
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ]
        if normalize:
            # 灰度用 CIFAR-10 三通道均值的平均近似
            gray_mean = sum((0.4914, 0.4822, 0.4465)) / 3.0
            gray_std = sum((0.2470, 0.2435, 0.2616)) / 3.0
            transform_list.append(transforms.Normalize((gray_mean,), (gray_std,)))
        transform = transforms.Compose(transform_list)
        D = 1024
    else:
        # 彩色图：3x32x32 = 3072 维
        transform_list = [
            transforms.ToTensor(),
        ]
        if normalize:
            # CIFAR-10 官方均值/方差（注意不是 CIFAR-100 的）
            transform_list.append(transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ))
        transform = transforms.Compose(transform_list)
        D = 3072

    # 下载并加载训练集和测试集（如果已经下载过，不会重复下载）
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )

    # 转换为张量堆叠
    x_train = torch.stack([img for img, _ in train_dataset])   # [50000, C, H, W]
    y_train = torch.tensor([label for _, label in train_dataset], dtype=torch.long)
    x_test = torch.stack([img for img, _ in test_dataset])
    y_test = torch.tensor([label for _, label in test_dataset], dtype=torch.long)

    # 展平（如果需要）
    if flatten:
        x_train = x_train.view(x_train.size(0), -1)   # [50000, D]
        x_test = x_test.view(x_test.size(0), -1)

    # 移动到指定设备
    if device != 'cpu':
        x_train = x_train.to(device)
        y_train = y_train.to(device)
        x_test = x_test.to(device)
        y_test = y_test.to(device)

    print(f"[CIFAR-10] Loaded: train {x_train.shape}, test {x_test.shape}, flattened dim={D}")
    return x_train, y_train, x_test, y_test