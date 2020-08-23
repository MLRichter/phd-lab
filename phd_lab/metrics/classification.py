from attr import attrs
from torch import Tensor
from typing import Union, Tuple
import torch


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
            correct_k = correct[:k].view(-1).float().sum(0)
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
