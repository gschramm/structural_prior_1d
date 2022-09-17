"""script for demonstrating deblurring and denoising in 1D with Bowsher's method"""
import abc
import numpy as np
import numpy.typing as npt


class LinearOperator(abc.ABC):

    def __init__(self, x_shape: tuple, y_shape: tuple) -> None:
        """Linear operator abstract base class that maps real array x to real array y

        Parameters
        ----------
        x_shape : tuple
            shape of x array
        y_shape : tuple
            shape of y array
        """
        super().__init__()

        self._x_shape = x_shape
        self._y_shape = y_shape

    @property
    def x_shape(self) -> tuple:
        """shape of x array

        Returns
        -------
        tuple
            shape of x array
        """
        return self._x_shape

    @property
    def y_shape(self):
        """shape of y array

        Returns
        -------
        tuple
            shape of y array
        """
        return self._y_shape

    @abc.abstractmethod
    def forward(self, x: npt.NDArray) -> npt.NDArray:
        """forward step

        Parameters
        ----------
        x : npt.NDArray
            x array

        Returns
        -------
        npt.NDArray
            the linear operator applied to x
        """
        pass

    @abc.abstractmethod
    def adjoint(self, y: npt.NDArray) -> npt.NDArray:
        """adjoint of forward step

        Parameters
        ----------
        y : npt.NDArray
            y array

        Returns
        -------
        npt.NDArray
            the adjoint of the linear operator applied to y
        """
        raise NotImplementedError()

    def adjointness_test(self) -> None:
        """test if adjoint is really the adjoint of forward
        """
        x = np.random.rand(*self._x_shape)
        y = np.random.rand(*self._y_shape)

        x_fwd = self.forward(x)
        y_back = self.adjoint(y)

        assert (np.isclose((x_fwd * y).sum(), (x * y_back).sum()))

    def norm(self, num_iter=20) -> float:
        """estimate norm of operator via power iterations

        Parameters
        ----------
        num_iter : int, optional
            number of iterations, by default 20

        Returns
        -------
        float
            the estimated norm
        """

        x = np.random.rand(*self._x_shape)

        for i in range(num_iter):
            x = self.adjoint(self.forward(x))
            n = np.linalg.norm(x.ravel())
            x /= n

        return np.sqrt(n)


class Convolution1D(LinearOperator):

    def __init__(self, n: int, kernel: npt.NDArray) -> None:
        super().__init__((n, ), (n, ))

        if (kernel.size % 2) != 1:
            raise ValueError('Kernel size must be odd')

        self._kernel = kernel
        self._half_kernel_size = self._kernel.size // 2
        self._mask = np.zeros(n, dtype=np.uint8)
        self._mask[self._half_kernel_size:-self._half_kernel_size] = 1

    def forward(self, x: npt.NDArray) -> npt.NDArray:
        tmp = np.convolve(x, self._kernel[::-1], mode='full')
        return tmp[self._half_kernel_size:-self._half_kernel_size] * self._mask

    def adjoint(self, y: npt.NDArray) -> npt.NDArray:
        tmp = np.convolve(y * self._mask, self._kernel, mode='full')
        return tmp[self._half_kernel_size:-self._half_kernel_size]


class BowsherGradient1D(LinearOperator):
    """1D Bowsher gradient operator"""

    def __init__(self, n: int, neighbors: tuple, num_nearest: int,
                 prior_array: npt.NDArray) -> None:
        """1D Bowsher gradient operator

        Parameters
        ----------
        n : int
            length of the 1D array input
        neighbors : tuple
            of ints containing the neighbors to consider - e.g. (-2,-1,1,2)
        num_nearest : int
            number of nearest neighbors
        prior_array : npt.NDArray
            structure prior input array used to determine the nearest neighbors
        """
        self._n = n

        self._prior_array = prior_array
        self._neighbors = neighbors
        self._num_nearest = num_nearest

        self._num_neighbors = len(self._neighbors)

        super().__init__((self._n, ), (self._n, self._num_neighbors))

        self._setup_mask()

        self._kernels = []
        for offset in self._neighbors:
            kernel = np.zeros(2 * abs(offset) + 1)
            kernel[abs(offset)] = -1
            kernel[abs(offset) + offset] = 1
            self._kernels.append(kernel)

    @property
    def prior_array(self) -> npt.NDArray:
        return self._prior_array

    @prior_array.setter
    def prior_array(self, p: npt.NDArray) -> None:
        self._prior_array = p
        self._setup_mask()

    @property
    def neighbors(self) -> tuple:
        return self._neighbors

    @neighbors.setter
    def neighbors(self, p: tuple) -> None:
        self._neighbors = p
        self._num_neighbors = len(self._neighbors)
        self._y_shape = (self._n, self._num_neighbors)
        self._setup_mask()

    @property
    def num_nearest(self) -> int:
        return self._num_nearest

    @num_nearest.setter
    def num_nearest(self, p: int) -> None:
        self._num_nearest = p
        self._setup_mask()

    @property
    def mask(self) -> npt.NDArray:
        return self._mask

    def _setup_mask(self) -> None:
        self._mask = np.zeros(self.y_shape, dtype=np.uint8)

        for i, s in enumerate(self._prior_array):
            diffs = []
            pos = []
            for j, offset in enumerate(self._neighbors):
                if ((i + offset) >= 0) and ((i + offset) < self._n):
                    diffs.append(abs(s - self._prior_array[i + offset]))
                    pos.append(j)

            self._mask[i,
                       np.array(pos)[np.argsort(np.array(diffs)
                                                )][:self._num_nearest]] = 1

    def forward(self, x: npt.NDArray) -> npt.NDArray:
        y = np.zeros(self.y_shape)
        for i, offset in enumerate(self._neighbors):
            tmp = np.convolve(x, self._kernels[i][::-1], mode='full')
            y[:, i] = tmp[abs(offset):-abs(offset)]
        return y * self._mask

    def adjoint(self, y: npt.NDArray) -> npt.NDArray:
        ym = y * self._mask
        x = np.zeros(self.x_shape)

        for i, offset in enumerate(self._neighbors):
            tmp = np.convolve(ym[:, i], self._kernels[i], mode='full')
            x += tmp[abs(offset):-abs(offset)]

        return x


if __name__ == '__main__':
    np.random.seed(1)

    n = 100
    neighbors = (-3, -2, -1, 1, 2, 3)
    num_nearest = 3
    p = np.random.rand(n)

    bg = BowsherGradient1D(n, neighbors, num_nearest, p)
    bg.adjointness_test()

    co = Convolution1D(n, np.random.rand(5))
    co.adjointness_test()
