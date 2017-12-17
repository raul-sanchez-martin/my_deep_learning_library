"""
A loss function measures how good or bad
our predictions are, and gives us gradient
"""

from .tensor import Tensor
import numpy as np

class Loss:
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError

    def grad(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError


class MSE(Loss):
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return np.sum((predicted - actual) ** 2)

    def grad(self, predicted: Tensor, actual: Tensor) -> float:
        return 2 * (predicted - actual)