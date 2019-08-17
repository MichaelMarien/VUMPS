from tensornetwork import TensorNetwork
import numpy as np


class MPS(TensorNetwork):

    def __init__(self, tensor, left_virtual=0, physical=1, right_virtual=2, backend="numpy"):
        super().__init__(backend)

        # Create the network
        axis_dict = {"left_virtual": left_virtual,
                     "right_virtual": right_virtual,
                     "physical": physical}

        self.axis_ordering = sorted(axis_dict, key=axis_dict.get, reverse=False)
        self.A = self.add_node(tensor, axis_names=self.axis_ordering)

    def _QRPos(self, L, A):
        raise NotImplementedError

    def left_orthonormalize(self, initial, epsilon):
        raise NotImplementedError

    @property
    def transfer_matrix(self):
        raise NotImplementedError

    @property
    def norm(self):
        raise NotImplementedError

    def apply_transfer_matrix(self):
        raise NotImplementedError

    def truncate(self):
        raise NotImplementedError
