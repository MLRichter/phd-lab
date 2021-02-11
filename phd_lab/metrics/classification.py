from pathlib import Path

from attr import attrs
from sklearn.base import ClassifierMixin
from sklearn.metrics import balanced_accuracy_score, plot_confusion_matrix
from matplotlib import pyplot as plt
from torch import Tensor
from typing import Union, Tuple, Optional
import torch
import numpy as np


@attrs(auto_attribs=True, slots=True)
class Top5Accuracy:

    accuracy_accumulator: Union[int, float, Tensor] = 0
    total: int = 0
    name: str = "top5-accuracy"

    def _accuracy(self, output: Tensor, target: Tensor, topk: Tuple[int] = (1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred).long())

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res

    @property
    def value(self) -> float:
        return self.accuracy_accumulator / self.total

    def update(self, y_true: Tensor, y_pred: Tensor) -> None:
        _, predicted = torch.max(y_pred.data, 1)
        self.accuracy_accumulator += self._accuracy(y_pred, y_true, (5,))[0]
        self.total += 1

    def reset(self) -> None:
        self.total = 0
        self.accuracy_accumulator = 0


@attrs(auto_attribs=True, slots=True)
class Accuracy:

    total: Union[int, float, Tensor] = 0
    correct: Union[int, float, Tensor] = 0
    name: str = "accuracy"

    @property
    def value(self) -> float:
        return 100 * (self.correct / self.total)

    def update(self, y_true: Tensor, y_pred: Tensor) -> None:
        _, predicted = torch.max(y_pred.data, 1)
        self.correct += (predicted == y_true.long()).sum().item()
        self.total += y_true.size(0)

    def reset(self) -> None:
        self.total = 0
        self.correct = 0


@attrs(auto_attribs=True, slots=True)
class BalancedAccuracy:

    y_true: Optional[np.ndarray] = None
    y_pred: Optional[np.ndarray] = None
    adjusted: bool = False
    name: str = "balanced accuracy"

    @property
    def value(self) -> float:
        return balanced_accuracy_score(self.y_true, self.y_pred, adjusted=self.adjusted)

    def update(self, y_true: Tensor, y_pred: Tensor) -> None:
        predicted = torch.max(y_pred.data, 1)[1].detach().cpu().numpy()
        ground_truth = y_true.detach().cpu().numpy()
        self.y_true = ground_truth if self.y_true is None else np.hstack((self.y_true, ground_truth))
        self.y_pred = predicted if self.y_pred is None else np.hstack((self.y_pred, predicted))

    def reset(self) -> None:
        self.y_true, self.y_pred = None, None


def AdjustedBalancedAccuracy():
    return BalancedAccuracy(adjusted=True, name="adjusted balanced accuracy")


@attrs(auto_attribs=True, slots=True)
class ConfusionMatrix:

    y_true: Optional[np.ndarray] = None
    y_pred: Optional[np.ndarray] = None
    step: int = 0
    adjusted: bool = False
    savepath: Path = Path("./logs/ConfusionMatrices")
    savename_template = "confusion_matrix_{}_{}"
    name: str = "balanced accuracy"

    @property
    def value(self) -> float:
        return (self.step) // 2

    def update(self, y_true: Tensor, y_pred: Tensor) -> None:
        predicted = torch.max(y_pred.data, 1)[1].detach().cpu().numpy()
        ground_truth = y_true.detach().cpu().numpy()
        self.y_true = ground_truth if self.y_true is None else np.hstack((self.y_true, ground_truth))
        self.y_pred = predicted if self.y_pred is None else np.hstack((self.y_pred, predicted))

    def reset(self) -> None:
        if self.y_true is not None:
            mode = "eval" if self.step%2 == 1 else "train"
            epoch = self.step // 2

            class DummyModel(ClassifierMixin):

                classes_ = np.unique(self.y_true)

                def predict(self, X):
                    return X
            plot_confusion_matrix(DummyModel(), self.y_pred, self.y_true, normalize="true")
            self.savepath.mkdir(exist_ok=True, parents=True)
            plt.savefig(self.savepath / self.savename_template.format(mode, epoch))
            self.step += 1
        self.y_true, self.y_pred = None, None