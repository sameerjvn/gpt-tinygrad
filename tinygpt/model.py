from tinygrad import nn
from tinygrad import Tensor


class SLP:
    def __init__(self, in_features: int, out_features: int):
        self.l1 = nn.Linear(in_features, out_features)

    def __call__(self, inputs: Tensor):
        return self.l1(inputs).sigmoid()


class MLP:
    def __init__(self, in_features: int, out_features: int):
        self.l1 = nn.Linear(in_features, in_features * 2)
        self.l2 = nn.Linear(in_features * 2, out_features)

    def __call__(self, inputs: Tensor):
        return self.l1(inputs).sigmoid()
