# states.py
from enum import Enum

class TreeState(Enum):
    HEALTHY = 0
    BURNING = 1
    BURNT = 2
