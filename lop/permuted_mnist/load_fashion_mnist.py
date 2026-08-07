import torch
import pickle
import torchvision
import torchvision.transforms as transforms

def fashion_mnist():
    batch_size = 60000
    transform = transforms.Compose([transforms.ToTensor()])

    # 关键修改1：使用 FashionMNIST 替代 MNIST
    train_dataset = torchvision.datasets.FashionMNIST(
        root="data", train=True, transform=transform, download=True
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        root="data", train=False, transform=transform
    )
    # Data loader
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )

    # flatten 维度保持一致，因为 Fashion-MNIST 图像也是 28*28=784
    for i, (images, labels) in enumerate(train_loader):
        images = images.flatten(start_dim=1)
        labels = labels
    x = images
    y = labels

    for i, (images_test, labels_test) in enumerate(test_loader):
        images_test = images_test.flatten(start_dim=1)
        labels_test = labels_test
    x_test = images_test
    y_test = labels_test

    # 关键修改2：单独保存为一个新文件，不要覆盖原始的 MNIST 数据
    with open('data/fashion_mnist_', 'wb+') as f:
        pickle.dump([x, y, x_test, y_test], f)

    return x, y, x_test, y_test

def get_fashion_mnist(type='reg'):
    if type == 'reg':
        data_file = 'data/fashion_mnist_'  # 匹配上面的文件名
        with open(data_file, 'rb+') as f:
            x, y, x_test, y_test = pickle.load(f)
    return x, y, x_test, y_test

if __name__ == '__main__':
    fashion_mnist()