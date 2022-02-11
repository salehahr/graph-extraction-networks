from enum import Enum
from typing import Optional

from .colours import *


class ColourEnum(Enum):
    def __new__(cls, *args, **kwds):
        value = len(cls.__members__)
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, _id: int, colour: Optional[tuple]):
        self._value_ = _id
        self.colour = colour


class NodePositions(ColourEnum):
    IS_NOT_NODE = (0, None)
    IS_NODE = (1, BGR_WHITE)


class NodeDegrees(ColourEnum):
    IS_NOT_NODE = (0, None)
    DEG1 = (1, BGR_WHITE)
    DEG2 = (2, BGR_GREEN)
    DEG3 = (3, BGR_RED)
    DEG4 = (4, BGR_BLUE)
    DEG5 = (5, BGR_LILAC)


class NodeTypes(ColourEnum):
    IS_NOT_NODE = (0, None)
    CROSSING = (1, BGR_BLUE)
    END = (2, BGR_RED)
    BORDER = (3, BGR_YELLOW)
