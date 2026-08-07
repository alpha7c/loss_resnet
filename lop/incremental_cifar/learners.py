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
    """

    def __init__(self, net: torch.nn.Module, optim: torch.optim.Optimizer,
                 loss_fn, device: torch.device):
        self.net = net
        self.optim = optim
        self.loss_fn = loss_fn
        self.device = device

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
                 maturity_threshold: int = 100):
        super().__init__(net, optim, loss_fn, device)
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
                 gamma: float = 0.93):
        super().__init__(net, optim, loss_fn, device)
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

    # filter kwargs to only pass relevant params to each learner
    if agent == "bp":
        return learner_cls(net=net, optim=optim, loss_fn=loss_fn, device=device)

    elif agent == "cbp":
        cbp_kwargs = {
            k: kwargs[k] for k in
            ["hidden_activation", "replacement_rate", "decay_rate",
             "util_type", "maturity_threshold"]
            if k in kwargs
        }
        return learner_cls(net=net, optim=optim, loss_fn=loss_fn,
                           device=device, **cbp_kwargs)

    elif agent == "ewc":
        ewc_kwargs = {
            k: kwargs[k] for k in
            ["ewc_lambda", "fisher_sample_size", "gamma"]
            if k in kwargs
        }
        return learner_cls(net=net, optim=optim, loss_fn=loss_fn,
                           device=device, **ewc_kwargs)
