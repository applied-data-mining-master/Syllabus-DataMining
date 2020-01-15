import numpy as np


class Perceptron(object):
    """Perceptron classifier. """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        """
        Perceptron classifier.
        
        :param eta: (float) Learning rate (between 0.0 and 1.0)
        :param n_iter: (int) Passes over the training dataset
        :param random_state: (int) Random number generator seed for random weight initialization.
        
        """
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        
        self.w_ = None
        self.errors_ = None
        
    def fit(self, x, y):
        """
        
        Fit training data.
        
        :param x: {array-like, shape=[n_samples, n_features]} Training vectors, where n_samples is the number of
        samples and n_features is the number of features.
        :param y: {array-like, shape=n_samples} Target values.
        :rtype: Perceptron
        :return: self
        
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + x.shape[1])
        
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(x, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self._errors.append(errors)
        return self
    
    def net_input(self, x):
        """
        
        Calculate net input
        
        :param x: {array-like, shape=[n_samples, n_features]} Training vectors, where n_samples is the number of
        samples and n_features is the number of features.
        :return:
        """
        
        return np.dot(x, self.w_[1:]) + self.w_[0]
    
    def predict(self, x):
        """
        Return class label after unit step
        
        :param x: {array-like, shape=[n_samples, n_features]} Training vectors, where n_samples is the number of
        samples and n_features is the number of features.
        :return:
        
        """
        return np.where(self.net_input(x) >= 0.0, 1, -1)