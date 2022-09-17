import abc
import numpy as np
import numpy.typing as npt

from operators import LinearOperator


class Norm(abc.ABC):
    """abstract base clase for norms where we can calculate the prox of the convex dual"""

    @abc.abstractmethod
    def __call__(self, x: npt.NDArray) -> float:
        """
        Parameters
        ----------
        x : npt.NDArray
            complex gradient of pseudo-complex array

        Returns
        -------
        float
            the complex gradient norm
        """
        raise NotImplementedError

    @abc.abstractmethod
    def prox_convex_dual(self, x: npt.NDArray, sigma: float) -> npt.NDArray:
        """proximal operator of the convex dual of the norm

        Parameters
        ----------
        x : npt.NDArray
            complex gradient of pseudo-complex array
        sigma : float, optional
            sigma parameter of prox, by default 1

        Returns
        -------
        npt.NDArray
            the proximity operator of the convex dual of the norm applied on x
        """
        raise NotImplementedError


class SmoothNorm(Norm):
    """smooth norm with gradient method"""

    @abc.abstractmethod
    def gradient(self, x: npt.NDArray) -> npt.NDArray:
        """gradient of norm

        Parameters
        ----------
        x : npt.NDArray
            input to norm

        Returns
        -------
        npt.NDArray
            the gradient
        """
        raise NotImplementedError


class ComplexL1L2Norm(Norm):
    """mixed L1-L2 norm of a pseudo-complex gradient field - real and imaginary part are treated separately"""

    def __init__(self) -> None:
        pass

    def __call__(self, x: npt.NDArray) -> float:
        n = np.linalg.norm(x[..., 0], axis=0).sum() + np.linalg.norm(
            x[..., 1], axis=0).sum()

        return n

    def prox_convex_dual(self, x: npt.NDArray, sigma: float) -> npt.NDArray:

        gnorm0 = np.linalg.norm(x[..., 0], axis=0)
        r0 = x[..., 0] / np.clip(gnorm0, 1, None)

        gnorm1 = np.linalg.norm(x[..., 1], axis=0)
        r1 = x[..., 1] / np.clip(gnorm1, 1, None)

        return np.stack([r0, r1], axis=-1)


class L2NormSquared(SmoothNorm):
    """squared L2 norm"""

    def __init__(self) -> None:
        pass

    def __call__(self, x: npt.NDArray) -> float:
        n = 0.5 * (x**2).sum()

        return n

    def prox_convex_dual(self, x: npt.NDArray, sigma: float) -> npt.NDArray:

        return x / (1 + sigma)

    def gradient(self, x: npt.NDArray) -> npt.NDArray:

        return x


class TotalCost:
    """ total (smooth) cost consisting of data fidelity and prior"""

    def __init__(self, data: npt.NDArray, data_operator: LinearOperator,
                 data_norm: SmoothNorm, prior_operator: LinearOperator,
                 prior_norm: SmoothNorm, beta: float) -> None:

        self._data = data

        self._data_operator = data_operator
        self._data_norm = data_norm

        self._prior_operator = prior_operator
        self._prior_norm = prior_norm

        self._beta = beta

    def __call__(self, x: npt.NDArray) -> float:
        input_shape = x.shape
        # reshaping is necessary since the scipy optimizers only handle 1D arrays
        x = x.reshape(self._data_operator.x_shape)

        cost = self._data_norm(self._data_operator.forward(x) -
                               self._data) + self._beta * self._prior_norm(
                                   self._prior_operator.forward(x))

        x = x.reshape(input_shape)

        return cost

    def gradient(self, x: npt.NDArray) -> npt.NDArray:
        input_shape = x.shape
        # reshaping is necessary since the scipy optimizers only handle 1D arrays
        x = x.reshape(self._data_operator.x_shape)

        data_grad = self._data_operator.adjoint(
            self._data_norm.gradient(
                self._data_operator.forward(x) - self._data))
        prior_grad = self._beta * self._prior_operator.adjoint(
            self._prior_norm.gradient(self._prior_operator.forward(x)))

        x = x.reshape(input_shape)

        return (data_grad + prior_grad).reshape(input_shape)
