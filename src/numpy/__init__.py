import math

float32 = float
pi = math.pi

class ndarray(list):
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return ndarray([x * other for x in self])
        return NotImplemented

    __rmul__ = __mul__

def zeros(shape, dtype=float32):
    if isinstance(shape, tuple):
        if len(shape) == 1:
            length = shape[0]
            return ndarray([dtype(0)] * length)
        elif len(shape) == 2:
            return ndarray([[dtype(0)] * shape[1] for _ in range(shape[0])])
        else:
            raise NotImplementedError
    else:
        return ndarray([dtype(0)] * int(shape))

def linspace(start, stop, num, endpoint=True):
    if num <= 1:
        return ndarray([start])
    step = (stop - start) / float(num - 1 if endpoint else num)
    values = [start + i * step for i in range(num)]
    if endpoint:
        values[-1] = stop
    return ndarray(values)

def sin(x):
    if isinstance(x, ndarray):
        return ndarray([math.sin(v) for v in x])
    return math.sin(x)

__all__ = [
    'ndarray', 'zeros', 'linspace', 'sin', 'pi', 'float32'
]
