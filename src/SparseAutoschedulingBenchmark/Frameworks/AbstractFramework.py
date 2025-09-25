from abc import ABC, abstractmethod
from .einsum import einsum


class AbstractFramework(ABC):
    def __init__(self, xp):
        self.xp = xp

    # Benchmark Format -> Eager Tensor
    @abstractmethod
    def from_benchmark(self, array):
        pass

    # Eager Tensor -> Benchmark Format
    @abstractmethod
    def to_benchmark(self, array):
        pass

    # Eager Tensor -> Lazy Tensor
    @abstractmethod
    def lazy(self, array):
        pass

    # Lazy Tensor -> Eager Tensor
    @abstractmethod
    def compute(self, array):
        pass

    def einsum(self, prgm, **kwargs):
        return einsum(self.xp, prgm, **kwargs)

    @abstractmethod
    def __getattr__(self, name):
        self.xp = xp
