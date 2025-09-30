__all__ = [
    "main",
    "einsum",
]
from .BenchmarkRunner import main as main
from .Frameworks.einsum import einsum
