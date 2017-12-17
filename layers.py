"""
our neural nets will made up of layer
one might look like


inputs --> linear --> tanh --> linear --> outputs
"""

from typing import Dict, Callable
from .tensor import Tensor
import numpy as np


class Layer:
    def __init__(self) -> None:
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}

    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError

    def backward(self, input: Tensor) -> Tensor:
        raise NotImplementedError


class Linear(Layer):
    """
    computes @ w + b
    """
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.params["w"] = np.random.randn(input_size, output_size)
        self.params["b"] = np.random.randn(output_size)

    def forward(self, inputs: Tensor) -> Tensor:
        # batch_size, input_size = inputs.shape
        self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"]

    def backward(self, grad: Tensor):
        """
        if y = f(x) and x = a*b
        then dy/da = f'(x) * b
        and dy/db = f'(x) * a

        now if x = a @ b
        then dy/da = f'(x) @ b.T
        and dy/db = a.T @ f'(x

        :param input:
        :return:
        """
        self.grads["b"] = np.sum(grad, axis=0)
        self.grads["w"] = self.inputs.T @ grad
        return grad @ self.params["w"].T


F = Callable[[Tensor], Tensor]


class Activation(Layer):
    def __init__(self, f:F, f_prime:F) -> None:
        self.f = f
        self.f_prime = f_prime

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return self.f(inputs)

    def backward(self, grad: Tensor):
        return self.f_prime(self.inputs) * grad


def tanh(x: Tensor) -> Tensor:
    return np.tanh(x)

def tanh_prime(x: Tensor) -> Tensor:
    y = tanh(x)
    return 1 - y ** 2

class Tanh(Activation):
    super().__init__(tanh, tanh_prime)