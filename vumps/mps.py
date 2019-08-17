from tensornetwork import TensorNetwork
import numpy as np


class MPS(TensorNetwork):

    axis_order = ["left_virtual", "physical", "right_virtual"]

    def __init__(self, tensor, left_virtual=0, physical=1, right_virtual=2,
                 backend="numpy"):
        super().__init__(backend)
        if tensor.shape[left_virtual] != tensor.shape[right_virtual]:
            raise ValueError("Left and right virtual dimension must be equal")
        # Create the network
        tensor = self.backend.transpose(tensor,
                                        [left_virtual, physical, right_virtual])
        self.A = self.add_node(tensor, axis_names=self.axis_order)
        self._virtual_dimension = self.backend.shape_tuple(self.A.tensor)[0]
        self._physical_dimension = self.backend.shape_tuple(self.A.tensor)[1]

    def _QRPos(self, L):
        new_shape = (self.virtual_dimension*self.physical_dimension,
                     self.virtual_dimension)
        la = np.dot(L, self.backend.reshape(self.A.tensor, new_shape))
        a_new, l_new = np.linalg.qr(la, mode="reduced")
        phases = 1j*np.angle(np.diag(a_new))
        return (np.dot(a_new, np.diag(np.exp(phases))),
                np.dot(l_new, np.diag(np.exp(-phases))))

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

    @property
    def virtual_dimension(self):
        return self._virtual_dimension

    @property
    def physical_dimension(self):
        return self._physical_dimension
