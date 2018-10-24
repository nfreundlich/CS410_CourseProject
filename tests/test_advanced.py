# -*- coding: utf-8 -*-

from .context import feature_mining

import unittest


class AdvancedTestSuite(unittest.TestCase):
    """Advanced test cases."""

    def test_advanced(self):
        self.assertIsNone(None)


if __name__ == '__main__':
    unittest.main()
