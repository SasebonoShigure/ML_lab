
import numpy as np
from typing import Tuple

class SVM():

    """
    Support Vector Machine model.
    """

    def __init__(
        self,
        kernel_fn,
    ) -> None:

        """
        Arguments:

        """

        self.kernel_fn = kernel_fn # Kernel function as one object in **kernels.py**
        self.b = None  # SVM's threshold, shape (1,)
        self.alpha = None  # SVM's dual variables, shape (n_support,)
        self.support_labels = None  # SVM's dual variables, shape (n_support,), in {-1, 1}
        self.support_vectors = None  # SVM's support vectors, shape (n_support, d)

    def predict(
        self,
        x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:

        """
        SVM predict method (dual form with some kernel).

        Arguments:
            x : (n, d), where n represents the number of samples, d the number of features

        Return:
            scores : (n,), SVM scores, where scores[i] is the score for x[i]
            pred : (n,), SVM predictions, where pred[i] is the prediction for x[i], in {-1, 1}
        """

        # TODO: implement predict method,
        # Asumme that: self.b, self.alpha, self.support_labels, self.support_vectors and self.kernel_fn are already given (They will be overwritten by SSMO optimization, which will be implemented in the next part)
        scores = self.support_labels * self.alpha @ self.kernel_fn(self.support_vectors, x) + self.b
        pred = np.where(scores >= 0, 1, -1)
        return scores, pred

# tests
# from kernels import *
# x = np.array([[1,0],[0,1],[0,0]])
# alpha = np.array([0.5,0.5,1])
# y = np.array([1,1,-1])
# x_test = np.array([[1,1],[-1,-1]])
# print(y*alpha)
# kernel = RBF_kernel(1)
# svm = SVM(kernel)
# K = kernel(x,x_test)
# print(K)
# scores = y*alpha@K
# print(scores)
# predict = np.where(scores >= 0, 1, -1)
# print(predict)