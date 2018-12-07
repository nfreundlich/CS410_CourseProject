# -*- coding: utf-8 -*-

from unittest import TestCase
from feature_mining.em_vector_by_feature import EmVectorByFeature
from feature_mining import ParseAndModel
import numpy as np
from scipy.sparse import csr_matrix


class TestEmVectorByFeature(TestCase):

    def test_constructor_twoline(self):
        pm_inst = ParseAndModel(feature_list=["screen"],
                                filename='../tests/data/parse_and_model/twoLineTest.txt')
        em = EmVectorByFeature(explicit_model=pm_inst)

        expected_section_word_counts_matrix = [[1, 1, 1, 0, 0]
            , [1, 0, 0, 1, 1]]
        expected_model_background_matrix = np.array([1 / 3, 1 / 6, 1 / 6, 1 / 6, 1 / 6])
        expected_model_feature_matrix = np.array([[0.218], [0.282], [0.282], [0.109], [0.109]])

        self.assertEqual(True,
                         np.array_equiv(expected_section_word_counts_matrix,
                                        csr_matrix.toarray(em.reviews_matrix)))
        self.assertEqual(True, np.array_equiv(expected_model_background_matrix,
                                              csr_matrix.toarray(em.background_probability)))
        self.assertEqual(True, np.array_equiv(expected_model_feature_matrix,
                                              np.round(em.topic_model, 3)))

        print("testing")

    def test_init_twoline(self):
        pm_inst = ParseAndModel(feature_list=["screen"],
                                filename='../tests/data/parse_and_model/twoLineTest.txt')
        em = EmVectorByFeature(explicit_model=pm_inst)

        em.initialize_parameters()

        pi_init = np.array([[1.],
                            [1.]])
        # em.pi_matrix.sum(axis=1)

        self.assertEqual(True, np.array_equiv(pi_init, em.pi_matrix))

    def test_e_step_twoline(self):
        pm_inst = ParseAndModel(feature_list=["screen"],
                                filename='../tests/data/parse_and_model/twoLineTest.txt')
        em = EmVectorByFeature(explicit_model=pm_inst)

        em.initialize_parameters()

        expected_hidden_feature = np.array([[1,1,1,0,0,],[1,0,0,1,1]])

        em.e_step()
        print("testing")

    def test_compute_denom(self):

        self.assertEqual(True, 1 < 0.001)

    def test_compute_nom(self):

        self.assertEqual(True, 1 < 0.001)

    def test_m_step(self):

        self.assertEqual(True, False)

