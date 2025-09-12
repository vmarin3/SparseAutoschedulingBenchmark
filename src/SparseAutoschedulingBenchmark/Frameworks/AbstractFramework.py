from abc import ABC, abstractmethod


class AbstractFramework(ABC):
    @abstractmethod
    def from_benchmark(self, array):
        pass

    @abstractmethod
    def to_benchmark(self, array):
        pass

    @abstractmethod
    def lazy(self, array):
        pass

    @abstractmethod
    def compute(self, array):
        pass

    @abstractmethod
    def __getattr__(self, name):
        pass
