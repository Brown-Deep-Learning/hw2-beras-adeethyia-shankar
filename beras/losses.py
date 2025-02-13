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
        return [Tensor(2 * np.mean(self.input_dict['y_pred'] -
                                   self.input_dict['y_true']))]

class CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true) -> Tensor:
        """Categorical cross entropy forward pass!"""
        return Tensor(-np.sum(np.dot(y_true, np.log(y_pred))))

    def get_input_gradients(self) -> list[Tensor]:
        """Categorical cross entropy input gradient method!"""
        # TODO: ensure 'y_pred' and 'y_true' are what I think they are
        return [Tensor(-np.sum(np.where(self.input_dict['y_pred'] == 0, 0,
                                        1 / self.input_dict['y_true'])))]