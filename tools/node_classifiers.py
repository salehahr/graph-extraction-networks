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
    IS_NODE = (1, bgr_white)


class NodeDegrees(ColourEnum):
    IS_NOT_NODE = (0, None)
    DEG1 = (1, bgr_white)
    DEG2 = (2, bgr_green)
    DEG3 = (3, bgr_red)
    DEG4 = (4, bgr_blue)
    DEG5 = (5, bgr_lilac)


class NodeTypes(ColourEnum):
    IS_NOT_NODE = (0, None)
    CROSSING = (1, bgr_blue)
    END = (2, bgr_red)
    BORDER = (3, bgr_yellow)
