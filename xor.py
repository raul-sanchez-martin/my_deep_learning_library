"""
Train a model to predict exclusive or of its inputs
"""

import numpy as np
from nn import NeuralNet
from layers import Linear, Tanh
from train import train

inputs = np.array([[0, 0],
                   [0, 1],
                   [1, 0],
                   [1, 1]])

targets = np.array([[1, 0],
                    [0, 1],
                    [0, 1],
                    [1, 0]])

net = NeuralNet([
    Linear(input_size=2, output_size=2),
    Tanh(),
    Linear(input_size=2, output_size=2)

])

train(net,
      inputs,
      targets,
      num_epochs = 5000)

for x, y in zip(inputs, targets):
    print(x, net.forward(x), y)