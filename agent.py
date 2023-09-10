# Standard Library
from enum import IntEnum


class Interest(IntEnum):
    RED = 2
    BLUE = 1


class Agent:
    def __init__(
        self,
        actv_temp: float = 1.0,
        pasv_temp: float = 1.0,
        alpha: float = 0.8,
        beta: float = 0.2,
    ) -> None:
        self.actv_temp = actv_temp
        self.pasv_temp = pasv_temp
        self.alpha = alpha
        self.beta = beta
