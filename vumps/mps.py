from tensornetwork import Node, TensorNetwork
import numpy as np
from scipy.sparse.linalg import LinearOperator


class MPS(Node):

    axis_order = ["left_virtual", "physical", "right_virtual"]
    ket_axis_order = ["ket_left_virtual", "ket_physical", "ket_right_virtual"]
    bra_axis_order = ["bra_left_virtual", "bra_physical", "bra_right_virtual"]

    def __init__(self, tensor, left_virtual=0, physical=1, right_virtual=2,
                 backend="numpy"):
        net = TensorNetwork(backend)

        if tensor.shape[left_virtual] != tensor.shape[right_virtual]:
            raise ValueError("Left and right virtual dimension must be equal")

        tensor = net.backend.transpose(tensor, [left_virtual, physical,
                                                right_virtual])

        super().__init__(axis_names=self.axis_order, network=net,
                         tensor=tensor, name="A")

        dimensions = self.network.backend.shape_tuple(self.tensor)
        self._virtual_dimension = dimensions[0]
        self._physical_dimension = dimensions[1]
        self.backend = backend

    def _QRPos(self, el):
        net = TensorNetwork(self.backend)
        a = net.add_node(self.tensor, axis_names=self.axis_order,
                         name="A")
        el = net.add_node(el, axis_names=["left", "right"], name="L")
        edge = net.connect(edge1=el["right"], edge2=a["left_virtual"])
        la = net.backend.reshape(net.contract(edge).tensor,
                                 np.array([self.virtual_dimension,
                                           self.physical_dimension,
                                           self.virtual_dimension]))
        la = MPS(la)

        mla = la.to_matrix(left_indices=["left_virtual", "physical"],
                           right_indices=["right_virtual"])
        a_new, l_new = np.linalg.qr(mla, mode="reduced")
        phases = 1j*np.angle(np.diag(a_new))
        return (np.dot(a_new, np.diag(np.exp(-phases))),
                np.dot(np.diag(np.exp(phases)), l_new))

    def left_orthonormalize(self, initial, epsilon):
        raise NotImplementedError

    def to_matrix(self, left_indices=None, right_indices=None):
        if len(left_indices + right_indices) != 3:
            raise ValueError("Please provide exactly 3 indices")
        if set(left_indices+right_indices) != set(self.axis_names):
            raise ValueError('Please provide all indices from {}'.format(
                self.axis_names))
        new_shape = (np.prod([self.get_dimension(x) for x in left_indices]),
                     np.prod([self.get_dimension(x) for x in right_indices]))
        permutation = [self.get_axis_number(x) for x in
                       left_indices + right_indices]
        new_tensor = self.network.backend.transpose(self.tensor, permutation)
        return self.network.backend.reshape(new_tensor, np.array(new_shape))

    @property
    def transfer_matrix(self):
        net = TensorNetwork(self.backend)
        ket = net.add_node(self.tensor, axis_names=self.ket_axis_order,
                           name="ket")
        bra = net.add_node(np.conj(self.tensor), axis_names=self.bra_axis_order,
                           name="bar")
        edge = net.connect(edge1=ket["ket_physical"], edge2=bra["bra_physical"])
        transfer_matrix = net.contract(edge)

        transfer_matrix.reorder_edges(edge_order=
                                      [ket.get_edge("ket_left_virtual"),
                                       bra.get_edge("bra_left_virtual"),
                                       ket.get_edge("ket_right_virtual"),
                                       bra.get_edge("bra_right_virtual")])
        return transfer_matrix

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

    def apply_transfer_matrix(self, v):
        return self.apply_mixed_transfer_matrix(np.conj(self.tensor), v)

    def apply_mixed_transfer_matrix(self, B, v):
        net = TensorNetwork(self.backend)
        ket = net.add_node(self.tensor, axis_names=self.ket_axis_order,
                           name="ket")
        bra = net.add_node(B, axis_names=self.bra_axis_order,
                           name="bar")
        vector = net.add_node(v, axis_names=["ket_virtual", "bra_virtual"],
                              name="vector")
        ket_edge = net.connect(edge1=ket["ket_left_virtual"],
                               edge2=vector["ket_virtual"])
        bra_edge = net.connect(edge1=bra["bra_left_virtual"],
                               edge2=vector["bra_virtual"])
        physical_edge = net.connect(edge1=ket["ket_physical"],
                                    edge2=bra["bra_physical"])
        net.contract(ket_edge)
        net.contract(bra_edge)
        result = net.contract(physical_edge)
        result.reorder_edges(edge_order=[ket.get_edge("ket_right_virtual"),
                                         bra.get_edge("bra_right_virtual")])
        return result.tensor
