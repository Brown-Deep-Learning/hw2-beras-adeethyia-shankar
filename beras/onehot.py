import numpy as np

from beras.core import Callable


class OneHotEncoder(Callable):
    """
    One-Hot Encodes labels. First takes in a candidate set to figure out what elements it
    needs to consider, and then one-hot encodes subsequent input datasets in the
    forward pass.

    SIMPLIFICATIONS:
     - Implementation assumes that entries are individual elements.
     - Forward will call fit if it hasn't been done yet; most implementations will just error.
     - keras does not have OneHotEncoder; has LabelEncoder, CategoricalEncoder, and to_categorical()
    """

    def fit(self, data):
        """
        Fits the one-hot encoder to a candidate dataset. Said dataset should contain
        all encounterable elements.

        :param data: 1D array containing labels.
            For example, data = [0, 1, 3, 3, 1, 9, ...]
        """
        unique_labels = np.unique(data)
        num_unique = len(unique_labels)
        self.onehot_dict = {unique_labels[i]: np.eye(num_unique)[i]
                            for i in range(num_unique)}
        self.inverse_dict = {v: k for k, v in self.onehot_dict.items()}

    def forward(self, data):
        return np.apply_along_axis(lambda row: self.onehot_dict[row], 1, data.reshape(-1, 1))

    def inverse(self, data):
        return np.apply_along_axis(lambda row: self.inverse_dict[row], 1, data).flatten()