import unittest

import numpy as np
from src.nmc import NMC
from src.data_perturb import CDataPerturbRandom


class TestNMC(unittest.TestCase):

    def SetUp(self):
        n_samples = 100
        n_features = 200
        self.x = np.zeros(shape=(n_samples, n_features))
        self.y = np.ones(shape=(n_samples,))
        self.clf = NMC()

    def test_init(self):
        "Check if centroids are None right after creation"
        self.assertTrue(self.clf.centroid is None)

    def test_fit(self):
        self.clf.fit(self.x, self.y)
        n_classes = np.unique(self.y).size
        expected_centroid_shape = (n_classes, self.x.shape[1])
        self.assertEqual(self.clf.centroid.shape, expected_centroid_shape)


    def test_predict(self):
        pass