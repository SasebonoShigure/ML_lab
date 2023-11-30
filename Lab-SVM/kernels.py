import numpy as np


class Base_kernel():

    def __init__(self):
        pass

    def __call__(self, x1, x2):
        """
        Linear kernel function.

        Arguments:
            x1: shape (n1, d)
            x2: shape (n2, d)

        Returns:
            y : shape (n1, n2), where y[i, j] = kernel(x1[i], x2[j])
        """
        pass


class Linear_kernel(Base_kernel):

    def __init__(self):
        super().__init__()

    def __call__(self, x1, x2):
        # TODO: Implement the linear kernel function
        y = x1.dot(x2.T)
        return y


class Polynomial_kernel(Base_kernel):

    def __init__(self, degree, c):
        super().__init__()
        self.degree = degree
        self.c = c

    def __call__(self, x1, x2):
        # TODO: Implement the polynomial kernel function
        inner_product = x1 @ x2.T
        y = (inner_product + self.c) ** self.degree
        return y

class RBF_kernel(Base_kernel):

    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma


    def __call__(self, x1, x2):
        # TODO: Implement the RBF kernel function
        if x1.ndim < 2:
            x1 = np.array([x1])
        if x2.ndim < 2:
            x2 = np.array([x2])
        diff = x1[:, np.newaxis, :] - x2[np.newaxis, :, :]
        norm = np.sum(diff ** 2, axis = 2)
        y = np.exp(- norm / (2 * self.sigma ** 2))
        return y

# tests
# x1 = np.array([[1,2],[3,4],[5,6]])
# x2 = np.array([3,4])
# kernel = RBF_kernel(2)
# print(kernel(x1,x2))

