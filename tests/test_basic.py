# -*- coding: utf-8 -*-

from .context import feature_mining

import unittest


class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""

    def test_basic(self):
        assert True


if __name__ == '__main__':
    unittest.main()