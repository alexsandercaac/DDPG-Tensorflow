"""

    Module to define classes associated with different action noise
    implementations.

"""
import abc
import numpy as np
from typing import Optional


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


class OUActionNoise(ActionNoise):
    """
        An Ornstein Uhlenbeck action noise, this is designed to approximate
        Brownian motion with friction. Noise is sampled from a correlated
        normal distribution.

        Implementation based on:
        https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/common/noise.py

        Args:

        mean (ndarray): the mean of the noise
        sigma (ndarray): the scale of the noise
        theta (ndarray): the rate of mean reversion. Default: 0.15.
        dt (float): the timestep for the noise. Default: 1e-2.
        initial_noise (ndarray): the initial value for the noise output.
            Default: None (0).

    """

    def __init__(
            self,
            mean: np.ndarray,
            sigma: np.ndarray,
            theta: float = 0.15,
            dt: float = 1e-2,
            initial_noise: Optional[np.ndarray] = None):

        super(OUActionNoise, self).__init__()
        self._theta = theta
        self._mu = mean
        self._sigma = sigma
        self._dt = dt
        self.initial_noise = initial_noise
        self.noise_prev = np.zeros_like(self._mu)
        self.reset()

    def __call__(self) -> np.ndarray:
        noise = (
            self.noise_prev
            + self._theta * (self._mu - self.noise_prev) * self._dt
            + self._sigma * np.sqrt(self._dt) *
            np.random.normal(size=self._mu.shape)
        )
        self.noise_prev = noise
        return noise

    def reset(self) -> None:
        """
        Reset the Ornstein Uhlenbeck noise to the initial position
        """
        self.noise_prev = self.initial_noise if self.initial_noise is not None\
            else np.zeros_like(self._mu)

    def __repr__(self) -> str:
        return (f"OrnsteinUhlenbeckActionNoise(mu={self._mu}, " +
                f"sigma={self._sigma})")


class GaussianActionNoise(ActionNoise):
    """
        Class for uncorrelated Gaussian action noise

        Args:

        mean (ndarray): the mean of the noise
        sigma (ndarray): the scale of the noise
    """
    def __init__(self, mean: np.ndarray, sigma: np.ndarray):

        self._mu = mean
        self._sigma = sigma
        super(GaussianActionNoise, self).__init__()

    def __call__(self) -> np.ndarray:

        noise = np.random.normal(loc=self._mu, scale=self._sigma,
                                 size=self._mu.shape)

        return noise

    def __repr__(self) -> str:
        return (f"GaussianActionNoise(mu={self._mu}," +
                f"sigma={self._sigma})")
