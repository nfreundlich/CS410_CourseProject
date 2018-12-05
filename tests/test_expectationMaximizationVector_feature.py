# -*- coding: utf-8 -*-

from unittest import TestCase
from feature_mining.em_vector_by_feature import EmVectorByFeature
import numpy as np
from scipy.sparse import csr_matrix


class TestExpectationMaximizationVector(TestCase):

    def test_initalization(self):
        em = em = EmVectorByFeature()
        print("testing")

    def test_e_step(self):

        self.assertEqual(True, False)

    def test_compute_denom(self):

        self.assertEqual(True, 1 < 0.001)

    def test_compute_nom(self):

        self.assertEqual(True, 1 < 0.001)

    def test_m_step(self):


        self.assertEqual(True, False)

