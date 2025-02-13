import numpy as np

from beras.core import Diffable, Tensor

import tensorflow as tf


class Loss(Diffable):
    @property
    def weights(self) -> list[Tensor]:
        return []

    def get_weight_gradients(self) -> list[Tensor]:
        return []


class MeanSquaredError(Loss):
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return Tensor(np.mean((y_pred - y_true)**2, keepdims=True))

    def get_input_gradients(self) -> list[Tensor]:
        # TODO: ensure I am accessing y_pred and y_true correctly
        return [Tensor(2 * np.mean(input[0] - input[1])) for input in self.inputs]

class CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        """Categorical cross entropy forward pass!"""
        return NotImplementedError

    def get_input_gradients(self):
        """Categorical cross entropy input gradient method!"""
        return NotImplementedError