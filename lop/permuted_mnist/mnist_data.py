# mnist_data.py
import torch
import torchvision
import torchvision.transforms as transforms


def load_mnist(device='cpu', flatten=True, normalize=False):
    """
    加载 MNIST 数据集，返回与 load_cifar10 接口兼容的张量。

    Args:
        device: 目标设备 ('cpu' 或 'cuda')
        flatten: 是否将图像展平为一维向量
        normalize: 是否用 MNIST 均值/方差归一化（ResNet 训练建议开启）。
                   注意：归一化在展平/像素排列之前完成，因此排列后的任务
                   数据依然保持归一化分布。

    Returns:
        x_train: torch.Tensor, shape [60000, D] 或 [60000, 1, 28, 28]
        y_train: torch.Tensor, shape [60000]
        x_test:  torch.Tensor, shape [10000, D] 或 [10000, 1, 28, 28]
        y_test:  torch.Tensor, shape [10000]
    """
    # MNIST: 1x28x28 = 784 维
    transform_list = [
        transforms.ToTensor(),
    ]
    if normalize:
        # MNIST 全局均值/方差
        transform_list.append(transforms.Normalize(
            mean=(0.1307,),
            std=(0.3081,),
        ))
    transform = transforms.Compose(transform_list)
    D = 784

    # 下载并加载训练集和测试集（如果已经下载过，不会重复下载）
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )

    # 转换为张量堆叠
    x_train = torch.stack([img for img, _ in train_dataset])   # [60000, 1, 28, 28]
    y_train = torch.tensor([label for _, label in train_dataset], dtype=torch.long)
    x_test = torch.stack([img for img, _ in test_dataset])
    y_test = torch.tensor([label for _, label in test_dataset], dtype=torch.long)

    # 展平（如果需要）
    if flatten:
        x_train = x_train.view(x_train.size(0), -1)   # [60000, 784]
        x_test = x_test.view(x_test.size(0), -1)

    # 移动到指定设备
    if device != 'cpu':
        x_train = x_train.to(device)
        y_train = y_train.to(device)
        x_test = x_test.to(device)
        y_test = y_test.to(device)

    print(f"[MNIST] Loaded: train {x_train.shape}, test {x_test.shape}, flattened dim={D}")
    return x_train, y_train, x_test, y_test
