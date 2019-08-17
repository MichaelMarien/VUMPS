from tensornetwork import TensorNetwork


class MPS(TensorNetwork):

    axis_order = ["left_virtual", "physical", "right_virtual"]

    def __init__(self, tensor, left_virtual=0, physical=1, right_virtual=2,
                 backend="numpy"):
        super().__init__(backend)

        # Create the network
        tensor = self.backend.transpose(tensor,
                                        [left_virtual, physical, right_virtual])
        self.A = self.add_node(tensor, axis_names=self.axis_order)

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
