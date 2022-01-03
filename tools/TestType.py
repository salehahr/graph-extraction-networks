from enum import Enum, unique


@unique
class TestType(Enum):
    TRAINING = 1
    VALIDATION = 2
    TESTING = 3
