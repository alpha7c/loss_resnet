import torch
import numpy as np
from lop.algos.bp import Backprop

class EWC(Backprop):
    def __init__(self, net, step_size, opt, loss, ewc_lambda=8,
                 weight_decay=0, device='cpu', fisher_sample_size=2000, **kwargs):
        super().__init__(
            net=net, step_size=step_size, opt=opt, loss=loss,
            weight_decay=weight_decay, device=device, **kwargs
        )
        self.ewc_lambda = ewc_lambda
        self.fisher_sample_size = fisher_sample_size
        self.fisher_matrix = None
        self.previous_params = None

    def _estimate_fisher(self, data_loader):
        """
        严格按照原论文，使用 '旧任务数据' 计算 Fisher 矩阵，量化参数对旧任务的重要性。
        """
        self.net.eval()
        fisher = {}
        # 初始化 Fisher 矩阵为零
        for name, param in self.net.named_parameters():
            fisher[name] = torch.zeros_like(param.data).to(self.device)

        total_samples = len(data_loader.dataset)
        # 随机采样部分数据来计算 Fisher 矩阵，提高效率
        sample_indices = np.random.choice(
            total_samples,
            min(self.fisher_sample_size, total_samples),
            replace=False
        )
        sampled_data = torch.utils.data.Subset(data_loader.dataset, sample_indices)
        sample_loader = torch.utils.data.DataLoader(
            sampled_data, batch_size=data_loader.batch_size, shuffle=False
        )

        n_samples = 0
        for data, target in sample_loader:
            data, target = data.to(self.device), target.to(self.device)
            self.opt.zero_grad()
            # 使用当前模型（旧任务的模型）在前向传播时，计算旧任务的损失
            output, _ = self.net.predict(data)
            loss = self.loss_func(output, target)
            loss.backward()

            for name, param in self.net.named_parameters():
                if param.grad is not None:
                    # 按批次大小加权，累加梯度平方
                    fisher[name] += param.grad.data.pow(2) * len(data)
            n_samples += len(data)

        # 求平均，得到最终的 Fisher 矩阵
        for name in fisher:
            fisher[name] /= n_samples

            fisher[name] = torch.clamp(fisher[name], max=0.01)
        self.net.train()
        return fisher

    def _ewc_loss(self):
        """计算 EWC 正则化项。"""
        loss = torch.tensor(0.0, device=self.device)
        if self.fisher_matrix is not None and self.previous_params is not None:
            for name, param in self.net.named_parameters():
                if name in self.fisher_matrix:
                    fisher = self.fisher_matrix[name]
                    prev_param = self.previous_params[name]
                    #loss = loss + (fisher * (param - prev_param).pow(2)).sum()
                    penalty = (param - prev_param).pow(2)

# ⭐ 防止 penalty 爆炸
                    penalty = torch.clamp(penalty, max=1.0)

                    loss = loss + (fisher * penalty).sum()
        return (self.ewc_lambda / 2) * loss

    def learn(self, x, target):
        """标准的学习步骤，加上 EWC 正则化项。"""
        x, target = x.to(self.device), target.to(self.device)
        self.opt.zero_grad()
        output, _ = self.net.predict(x)
        task_loss = self.loss_func(output, target)
        ewc_loss = self._ewc_loss()
        total_loss = task_loss + ewc_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=100.0)
        self.opt.step()
        if self.to_perturb:
            self.perturb()
        if self.loss == 'nll':
            return total_loss.detach(), output.detach()
        return total_loss.detach()

    def after_task(self, data_loader):
        """关键函数：任务结束后，将当前任务的 Fisher 矩阵累加到总矩阵中，并保存参数副本。"""
        print(f"  -> Computing Fisher matrix with {self.fisher_sample_size} samples...")
        new_fisher = self._estimate_fisher(data_loader)

        gamma = 0.93  # ⭐关键参数（几乎不忘）

        if self.fisher_matrix is None:
            self.fisher_matrix = new_fisher
        else:
            # 这里是累加，而非覆盖，用于保护所有历史任务
            for name in self.fisher_matrix:
                self.fisher_matrix[name] = (
                    gamma * self.fisher_matrix[name]
                    + new_fisher[name]
                )

# ⭐ 再限制一次总 Fisher
                self.fisher_matrix[name] = torch.clamp(
                    self.fisher_matrix[name],
                    max=0.01
                )
                #self.fisher_matrix[name] = gamma * self.fisher_matrix[name] + new_fisher[name]
                # self.fisher_matrix[name] += new_fisher[name]

        # 保存当前任务的参数，作为未来任务的锚点
        self.previous_params = {name: param.data.clone() for name, param in self.net.named_parameters()}
        fisher_norm = sum(t.norm().item() for t in self.fisher_matrix.values())
        print(f"  -> Accumulated Fisher matrix norm: {fisher_norm:.6f}")