from math import sqrt

class Tensor(list):
    def __init__(self, data):
        if isinstance(data, Tensor):
            data = data.data
        super().__init__(data)

    @property
    def shape(self):
        def _shape(d):
            if isinstance(d, list):
                if not d:
                    return (0,)
                return (len(d),) + _shape(d[0])
            else:
                return ()
        return _shape(self)

    def unsqueeze(self, dim):
        if dim == 0:
            return Tensor([self])
        raise NotImplementedError

    def float(self):
        return self

    def to(self, device):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self

    def mean(self, dim=None):
        if dim is None:
            return sum(self)/len(self)
        if dim == -1 or dim == len(self.shape)-1:
            return Tensor([sum(row)/len(row) for row in self])
        raise NotImplementedError

    def abs(self):
        return Tensor([abs(x) for x in self])

    def max(self):
        m = self[0]
        for x in self:
            if x > m:
                m = x
        return m

    def __truediv__(self, other):
        return Tensor([x/other for x in self])

    def __mul__(self, other):
        return Tensor([x*other for x in self])

    __rmul__ = __mul__

    def repeat(self, *sizes):
        data = self
        for s in reversed(sizes):
            data = [data for _ in range(s)]
        return Tensor(data)


float32 = float


def tensor(data, dtype=None):
    return Tensor(list(data))


def zeros(*shape):
    if len(shape) == 1:
        return Tensor([0.0]*shape[0])
    elif len(shape) == 2:
        return Tensor([[0.0]*shape[1] for _ in range(shape[0])])
    elif len(shape) == 3:
        return Tensor([[[0.0]*shape[2] for _ in range(shape[1])] for _ in range(shape[0])])
    else:
        raise NotImplementedError


def stack(tensors):
    return Tensor([list(t) for t in zip(*tensors)])


class _NNFunctional:
    @staticmethod
    def pad(tensor, pad):
        left, right = pad
        return Tensor(tensor + [0.0]*right)


class no_grad:
    def __enter__(self):
        pass
    def __exit__(self, exc_type, exc, tb):
        pass


nn = type('nn', (), {'functional': _NNFunctional()})

__all__ = ['Tensor', 'tensor', 'zeros', 'stack', 'float32', 'nn', 'no_grad']
