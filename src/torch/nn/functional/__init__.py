from ... import Tensor


def pad(tensor, pad):
    left, right = pad
    return Tensor(tensor + [0.0] * right)

