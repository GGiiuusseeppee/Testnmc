import unittest
from nmc import NMC
import numpy as np


class TestNMC(unittest.TestCase):

    def setUp(self):
        n_samples = 100
        n_features = 20
        self.x = np.zeros(shape=(n_samples, n_features))
        self.y = np.zeros(shape=(n_samples,))
        self.y[50:] = 1
        self.clf = NMC()

    def test_init(self):
        """Check if centroids are None right after creation."""
        self.assertTrue(self.clf.centroid is None)

    def test_robust_estimation(self):
        # self.clf.robust_est = 'sdjksakli'  # not boolean type
        self.assertRaises(TypeError, setattr, self.clf, 'robust_est',  'sdjksakli')

    def test_fit(self):
        expected_centroid_shape = (2, self.x.shape[1])
        self.clf.fit(self.x, self.y)
        self.assertEqual(self.clf.centroid.shape, expected_centroid_shape)

    def test_predict(self):
        pass
