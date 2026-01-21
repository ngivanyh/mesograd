from enum import Enum
from math import tanh, exp
from typing import Tuple

class _Act(Enum):
    relu = "ReLU"
    tanh = "tanh"
    sigmoid = "sigmoid"

class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data: int|float, _children=(), _op: str='', _act: _Act=_Act.relu):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc
        self._act = _act

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out
    
    def _relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out
    
    def _tanh(self):
        out = Value(tanh(self.data), (self,), "tanh")
        
        def _backward():
            self.grad += 0
        out._backward = _backward
        
        return out
    
    def _sigmoid(self):
        out = Value(1 / 1 + exp(-self.data), (self,), "Sigmoid")
        
        def _backward():
            self.grad += 0
        out._backward = _backward
        
        return out
    
    def act(self):
        # activation function, defaults to ReLU
        match self._act.value:
            case "tanh":
                return self._tanh()
            case "sigmoid":
                return self._sigmoid
            case _:
                return self._relu()

    def backward(self):
        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

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
        return f"Value(data={self.data}, grad={self.grad})"

class Tensor:
    """extends the Value object and groups them into a Tensor"""

    def __init__(self, data: int|float, dimensions: Tuple[int, int]=(1,1), _children=(), _op:str='', _act:_Act=_Act.relu):
        pass
    
    def backward(self):
        pass