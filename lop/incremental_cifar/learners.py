"""
Unified Learner classes for the ResNet continual learning experiment.

Each Learner encapsulates a specific algorithm's forward/backward/update logic,
exposing a clean `learn(images, labels)` interface to the training loop.

Supported algorithms:
    - bp  : standard SGD backpropagation
    - cbp : Continual Backprop (neuron generate-and-test)
    - ewc : Elastic Weight Consolidation (Fisher regularization)

To add a new algorithm:
    1. Subclass BaseResNetLearner (or an existing Learner)
    2. Implement learn(images, labels) → (loss, predictions)
    3. Optionally override after_task(dataloader) for post-task hooks
    4. Add the Learner to the agent_map in the experiment class
"""

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from lop.algos.res_gnt import ResGnT
from lop.algos.resnet_ewc import ResNetEWC


class BaseResNetLearner:
    """
    Base class for all ResNet learners.

    :param net: ResNet model
    :param optim: torch optimizer (SGD)
    :param loss_fn: loss function (e.g. CrossEntropyLoss)
    :param device: torch device
    :param weight_decay: 解耦 weight decay 系数；优化器自身的 weight_decay 必须为 0，
                         衰减由 _apply_decoupled_weight_decay 在 step 之后显式施加。
    """

    def __init__(self, net: torch.nn.Module, optim: torch.optim.Optimizer,
                 loss_fn, device: torch.device, weight_decay: float = 0.0):
        self.net = net
        self.optim = optim
        self.loss_fn = loss_fn
        self.device = device
        self.weight_decay = weight_decay

    # ------------------------------------------------------------------ #
    def _apply_decoupled_weight_decay(self):
        """
        解耦 weight decay（AdamW 语义）：在 optim.step() 之后把参数按 (1 - lr * wd) 收缩。

        为什么要解耦：
          把 (λ/2)‖θ‖² 写进 loss（= 优化器的 weight_decay）时，衰减项会先进入动量缓冲，
          稳态被放大 1/(1-momentum) 倍；本项目 momentum=0.9 即 10 倍，实际正则强度远大于
          名义值。解耦后每步精确收缩 lr*wd，没有放大。
          两者固定点相同（∇CE + wd*θ = 0），所以 λ 在目标函数中的语义不变。

        只衰减 ndim > 1 的参数（卷积/全连接权重）：BN 的 γ/β 与所有 bias 不衰减，
        这是标准 ResNet 配方的做法（BN 网络对权重尺度不敏感，衰减 γ/β 只会缩小
        各层输出尺度、间接压低有效学习率）。
        """
        if self.weight_decay <= 0:
            return
        lr = self.optim.param_groups[0]["lr"]   # set_lr() 会改它，不能缓存
        if lr == 0:
            return
        factor = 1.0 - lr * self.weight_decay
        with torch.no_grad():
            for p in self.net.parameters():
                if p.ndim > 1:
                    p.mul_(factor)

    # ------------------------------------------------------------------ #
    def learn(self, images: Tensor, labels: Tensor):
        """
        Execute one training step: forward, loss, backward, update.

        Subclasses MUST override this method.

        :param images: batch of images (B, 3, H, W)
        :param labels: batch of labels (B, num_classes) one-hot
        :return: (loss, predictions) — both detached
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    def after_task(self, dataloader: DataLoader):
        """
        Post-task hook. Called after a task's training is complete.
        Override in subclasses that need task-level bookkeeping (e.g. EWC).

        :param dataloader: DataLoader for the just-finished task
        """
        pass

    # ------------------------------------------------------------------ #
    def get_state(self) -> dict:
        """Return serializable state for checkpointing."""
        return {}

    # ------------------------------------------------------------------ #
    def load_state(self, state: dict):
        """Restore learner state from checkpoint."""
        pass


# ====================================================================== #
class BPLearner(BaseResNetLearner):
    """Standard SGD backpropagation."""

    def learn(self, images: Tensor, labels: Tensor):
        predictions = self.net.forward(images)
        loss = self.loss_fn(predictions, labels)
        loss.backward()
        self.optim.step()
        self._apply_decoupled_weight_decay()
        return loss.detach().clone(), predictions.detach()


# ====================================================================== #
class CBPLearner(BaseResNetLearner):
    """
    Continual Backprop: after each gradient step, low-utility neurons are
    replaced via the Generate-and-Test algorithm (ResGnT).
    """

    def __init__(self, net, optim, loss_fn, device,
                 hidden_activation: str = "relu",
                 replacement_rate: float = 1e-4,
                 decay_rate: float = 0.99,
                 util_type: str = "weight",
                 maturity_threshold: int = 100,
                 weight_decay: float = 0.0):
        super().__init__(net, optim, loss_fn, device, weight_decay=weight_decay)
        self.resgnt = ResGnT(
            net=net,
            hidden_activation=hidden_activation,
            replacement_rate=replacement_rate,
            decay_rate=decay_rate,
            util_type=util_type,
            maturity_threshold=maturity_threshold,
            device=device,
        )
        self._current_features = []  # reused across steps to avoid reallocation

    def learn(self, images: Tensor, labels: Tensor):
        # collect intermediate features for CBP
        self._current_features.clear()
        predictions = self.net.forward(images, self._current_features)
        loss = self.loss_fn(predictions, labels)
        loss.backward()
        self.optim.step()
        self._apply_decoupled_weight_decay()
        self.resgnt.gen_and_test(self._current_features)
        return loss.detach().clone(), predictions.detach()

    def get_state(self) -> dict:
        return {"resgnt": self.resgnt}

    def load_state(self, state: dict):
        if "resgnt" in state:
            self.resgnt = state["resgnt"]


# ====================================================================== #
class EWCLearner(BaseResNetLearner):
    """
    Elastic Weight Consolidation: adds a Fisher-weighted regularization term
    to the loss to protect parameters that were important for previous tasks.
    """

    def __init__(self, net, optim, loss_fn, device,
                 ewc_lambda: float = 5000,
                 fisher_sample_size: int = 2000,
                 gamma: float = 0.93,
                 weight_decay: float = 0.0):
        super().__init__(net, optim, loss_fn, device, weight_decay=weight_decay)
        self.ewc = ResNetEWC(
            net=net,
            ewc_lambda=ewc_lambda,
            device=device,
            fisher_sample_size=fisher_sample_size,
            gamma=gamma,
        )

    def learn(self, images: Tensor, labels: Tensor):
        predictions = self.net.forward(images)
        task_loss = self.loss_fn(predictions, labels)
        ewc_loss = self.ewc.penalty()
        total_loss = task_loss + ewc_loss
        total_loss.backward()
        self.optim.step()
        self._apply_decoupled_weight_decay()
        return total_loss.detach().clone(), predictions.detach()

    def after_task(self, dataloader: DataLoader):
        self.ewc.after_task(dataloader, self.loss_fn)

    def get_state(self) -> dict:
        return {"ewc": self.ewc.get_state()}

    def load_state(self, state: dict):
        if "ewc" in state:
            self.ewc.load_state(state["ewc"])


# ====================================================================== #
# Factory
# ====================================================================== #
def build_learner(agent: str, net: torch.nn.Module, optim: torch.optim.Optimizer,
                  loss_fn, device: torch.device, **kwargs) -> BaseResNetLearner:
    """
    Factory function to create a learner from a string identifier.

    :param agent: one of "bp", "cbp", "ewc"
    :param net: ResNet model
    :param optim: torch optimizer
    :param loss_fn: loss function
    :param device: torch device
    :param kwargs: algorithm-specific parameters (see each Learner's __init__)
    :return: BaseResNetLearner instance
    """
    agent_map = {
        "bp": BPLearner,
        "cbp": CBPLearner,
        "ewc": EWCLearner,
    }

    if agent not in agent_map:
        raise ValueError(
            "Unknown agent '{0}'. Supported: {1}".format(
                agent, list(agent_map.keys())
            )
        )

    learner_cls = agent_map[agent]

    # 解耦 weight decay 系数（优化器自身的 weight_decay 必须为 0，见 BaseResNetLearner）
    weight_decay = float(kwargs.get("weight_decay", 0.0))

    # filter kwargs to only pass relevant params to each learner
    if agent == "bp":
        return learner_cls(net=net, optim=optim, loss_fn=loss_fn, device=device,
                           weight_decay=weight_decay)

    elif agent == "cbp":
        cbp_kwargs = {
            k: kwargs[k] for k in
            ["hidden_activation", "replacement_rate", "decay_rate",
             "util_type", "maturity_threshold"]
            if k in kwargs
        }
        return learner_cls(net=net, optim=optim, loss_fn=loss_fn,
                           device=device, weight_decay=weight_decay, **cbp_kwargs)

    elif agent == "ewc":
        ewc_kwargs = {
            k: kwargs[k] for k in
            ["ewc_lambda", "fisher_sample_size", "gamma"]
            if k in kwargs
        }
        return learner_cls(net=net, optim=optim, loss_fn=loss_fn,
                           device=device, weight_decay=weight_decay, **ewc_kwargs)
