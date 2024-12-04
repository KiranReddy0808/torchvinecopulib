from . import bicop, vinecop, util

__all__ = [
    "bicop",
    "util",
    "vinecop",
]

import importlib.metadata

__version__ = importlib.metadata.version("torchvinecopulib")
