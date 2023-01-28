"""

    Module to define classes associated with different action noise
    implementations.

"""
import abc
import numpy as np


class ActionNoise(metaclass=abc.ABCMeta):
    """
    The action noise base class, used as a blueŕint for all action noise class
    implementations.
    """

    def __init__(self):
        super(ActionNoise, self).__init__()

    @abc.abstractmethod
    def reset(self) -> None:
        """
        Call at end of episode to reset noise
        """
        pass

    @abc.abstractmethod
    def __call__(self) -> np.ndarray:
        raise NotImplementedError()
