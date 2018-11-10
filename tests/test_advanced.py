# -*- coding: utf-8 -*-

from .context import feature_mining
from feature_mining import ExpectationMaximization
from feature_mining import ExpectationMaximizationOriginal
from feature_mining import ExpectationMaximizationVector
import numpy as np

import unittest


class AdvancedTestSuite(unittest.TestCase):
    """Advanced test cases."""

    def test_advanced_setup(self):
        import os
        print("Advanced test setup.")
        print("CWD:")
        print(os.getcwd())
        self.assertIsNone(None)

    def test_em_creation(self):
        em = feature_mining.ExpectationMaximizationOriginal()
        em = feature_mining.ExpectationMaximizationVector()
        em = ExpectationMaximizationOriginal()
        em = ExpectationMaximizationVector()
        self.assertIsNone(None)


if __name__ == '__main__':
    unittest.main()
