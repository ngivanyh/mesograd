from enum import Enum
from math import tanh, exp
from typing import Tuple, List

class Activations(Enum):
    relu = "ReLU"
    tanh = "tanh"
    sigmoid = "Sigmoid"
    linear = "Linear"

class _Value:
    """skeleton for other data types (Scalar, Vector, and Matrix)"""

    def __init__(
        self,
        data: int|float,
        _children=(),
        _op: str='',
        _act: Activations = Activations.relu
    ):
        """
        internal variables:
            - Used for autograph construction:
                - _backward: function that does the grad calc
                - _prev: previous node(s)
                - _op: the op that produced self (this node)
            - _act: customizable activation function (tanh, sigmoid, or relu)
        """
        self.data = data
        self.grad = None
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc
        self._act = _act

    def act(self):
        # activation function, default set to ReLU, linear if something happens to self._act.value
        match self._act.value:
            case "tanh":
                return self._tanh()
            case "Sigmoid":
                return self._sigmoid()
            case "ReLU":
                return self._relu()
            case _:
                return self # aka Linear

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"{self.__class__.__name__}(data={self.data}, grad={self.grad})"


class Scalar(_Value):
    """stores a single scalar value and its gradient"""

    def __init__(
        self,
        data: int|float,
        _children=(),
        _op: str='',
        _act: Activations = Activations.relu
    ):
        super().__init__(data, _children, _op, _act)
        self.grad = 0

    # the bases of every other operation
    def __add__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Scalar(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    # the three activation functions, .act() not need to be redefined
    def _relu(self):
        out = Scalar(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def _tanh(self):
        out = Scalar(tanh(self.data), (self,), "tanh")

        def _backward():
            # 1 - tanh(x)^2
            self.grad += (1 - out.data**2) * out.grad
        out._backward = _backward

        return out

    def _sigmoid(self):
        out = Scalar(1 / (1 + exp(-self.data)), (self,), "Sigmoid")

        def _backward():
            # σ(x)*σ(1 - x)
            self.grad += (out.data * (1 - out.data)) * out.grad
        out._backward = _backward

        return out

    def backward(self):
        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v in visited: return
            # v is not in visited
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)

        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()


class Vector(_Value):
    """stores a vector and its gradients"""

    def __init__(self,
        data: Tuple[int|float, ...] | List[int|float],
        _children=(),
        _op:str='',
        _act: Activations = Activations.relu
    ):
        super().__init__(data, _children, _op, _act)
        self.grad = [0.0 for _ in self.data]

    def __add__(self):
        pass

    def __mul__(self):
        pass

    def __pow__(self):
        pass

    def _tanh(self):
        pass

    def _relu(self):
        pass

    def _sigmoid(self):
        pass

    def backward(self):
        pass

    # a nice touch
    def __iter__(self):
        pass

    def __contains__(self):
        pass


# class Matrix(_Value):
#     """stores a matrix and its gradients"""

#     def __init__(self,
#         data: int|float,
#         dimensions: Tuple[int, int]=(1,1),
#         _children=(),
#         _op:str='',
#         _act: Activations = Activations.relu
#     ):
#         super().__init__(data, _children, _op, _act)
#         self.grad = 0

#     def __add__(self):
#         pass

#     def __mul__(self):
#         pass

#     def __pow__(self):
#         pass

#     def _tanh(self):
#         pass

#     def _relu(self):
#         pass

#     def _sigmoid(self):
#         pass

#     def backward(self):
#         pass