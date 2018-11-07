# -*- coding: utf-8 -*-

from .context import feature_mining

import unittest


class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""

    def test_basic(self):
        assert True

    def test_pass(self):
        self.assertEqual(3, 3)
        self.assertEqual(4, 4)

    def test_fail(self):
        self.assertEqual(3, 2)


if __name__ == '__main__':
    unittest.main()