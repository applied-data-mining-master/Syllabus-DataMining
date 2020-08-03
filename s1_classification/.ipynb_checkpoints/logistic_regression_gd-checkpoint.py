import numpy as np


class LogisticRegressionGD(object):
    """
    Logistic Regression Classifier using gradient descent.
    """
    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

        self._w = None
        self._cost = None

    def fit(self, x, y):
        rgen = np.random.RandomState(self.random_state)
        self._w = rgen.normal(loc=0.0, scale=0.01, size=1 + x.shape[1])
        self._cost = []

        for i in range(self.n_iter):
            net_input = self.net_input(x)
            output = self.activation(net_input)
            errors = (y - output)
            self._w[1:] += self.eta * x.T.dot(errors)
            self._w[0] += self.eta * errors.sum()

            # note that we compute the logistic cost now
            # instead of the sum of squared errors cost
            cost = (-y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output))))
            self._cost.append(cost)
        return self

    def net_input(self, x):
        """
        Calculate net input
        :param x:
        :return:
        """
        return np.dot(x, self._w[1:]) + self._w[0]

    def activation(self, z):
        """
        Compute logistic sigmoid activation
        :param z:
        :return:
        """
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, x):
        """
        Return class label after unit step.
        Equivalent to:
        >>> return np.where(self.activation(self.net_input(x)) > 0.5, 1, 0)
        :param x:
        :return:
        """
        return np.where(self.net_input(x) >= 0.0, 1, 0)
