import unittest

import numpy as np

from src.nmc import NMC


class TestNMC(unittest.TestCase):

    def SetUp(self):
        n_samples = 100
        n_features = 200
        self.x = np.zeros(shape=(n_samples, n_features))
        self.y = np.zeros(shape=(n_samples,))

    def test_fit(self):
        pass

    def test_predict(self):
        pass