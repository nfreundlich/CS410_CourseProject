# -*- coding: utf-8 -*-

from .context import feature_mining
from feature_mining import EM
from feature_mining import ExpectationMaximization
from feature_mining import ExpectationMaximizationOriginal
from feature_mining import ExpectationMaximizationVector
import numpy as np

import unittest
from unittest import TestSuite


class AdvancedTestSuite(unittest.TestCase):
    """Advanced test cases."""

    def test_advanced_setup(self):
        import os
        print("Advanced test setup.")
        print("CWD:")
        print(os.getcwd())
        self.assertIsNone(None)

    def test_em_vector_e_step_hp_01(self):
        dump_path = "./tests/data/em_01/"
        em = ExpectationMaximizationVector(dump_path=dump_path)
        em.em()

        hp_updated_by_santu = np.load(dump_path + "HP_updated.npy")
        hp_em_vector_one_sentence_for_testing = em.hidden_parameters_one_sentence_for_testing

        print("Values computed by e_step_vector:")
        sentence = 0
        print(em.aspects_map.keys())
        for i in np.where(em.reviews_matrix[sentence].todense() > 0)[1]:
            print(em.words_list[i], hp_em_vector_one_sentence_for_testing[i])
        print("Values computed by e_step_original")
        for key in hp_updated_by_santu[0][0]:
            print(key, hp_updated_by_santu[0][0][key])

        first_step_ok = True
        aspects_list = []
        for k, v in em.aspects_map.items():
            aspects_list.append(k)
        for i in np.where(em.reviews_matrix[sentence].todense() > 0)[1]:
            print(em.words_list[i])
            for j in range(0, len(np.array(hp_em_vector_one_sentence_for_testing[i]).squeeze())):
                print(aspects_list[j], np.array(hp_em_vector_one_sentence_for_testing[i]).squeeze()[j])
                print(hp_updated_by_santu[0][0][em.words_list[i]][aspects_list[j]])
                if np.array(hp_em_vector_one_sentence_for_testing[i]).squeeze()[j] - \
                        hp_updated_by_santu[0][0][em.words_list[i]][aspects_list[j]] > 0.001:
                    first_step_ok = False
                    break
        self.assertEqual(True, first_step_ok)

    def test_em_vector_e_step_bhp_01(self):
        dump_path = "./tests/data/em_01/"
        em = ExpectationMaximizationVector(dump_path=dump_path)
        em.em()

        hp_background_em_vector = em.hidden_parameters_background_one_sentence_for_testing
        hpb_updated_by_expectation_minimization_original = np.load(dump_path + "MY_HPB_Updated.npy")

        background_one_sentence_ok = True
        for i in np.where(hp_background_em_vector > 0)[0]:
            print(em.words_list[i], hp_background_em_vector[i].item())
            print(hpb_updated_by_expectation_minimization_original[0][0][em.words_list[i]])
            if hp_background_em_vector[i].item() - hpb_updated_by_expectation_minimization_original[0][0][em.words_list[i]] > 0.001:
                background_one_sentence_ok = False

        self.assertEqual(True, background_one_sentence_ok)

    def test_em_original_e_step_hp(self):
        dump_path = "./tests/data/em_01/"

        em = ExpectationMaximizationOriginal(dump_path=dump_path)
        em.em()
        em._dump_hidden_parameters()

        hp_updated_by_santu = np.load(dump_path + "HP_updated.npy")
        hp_updated_by_expectation_minimization_original = np.load(dump_path + "MY_HP_Updated.npy")

        self.assertEqual(hp_updated_by_santu.all(), hp_updated_by_expectation_minimization_original.all())

    def test_em_original_e_step_hpb(self):
        dump_path = "./tests/data/em_01/"

        em = ExpectationMaximizationOriginal(dump_path=dump_path)
        em.em()
        em._dump_hidden_parameters()

        hpb_updated_by_santu = np.load(dump_path + "HPB_updated.npy")
        hpb_updated_by_expectation_minimization_original = np.load(dump_path + "MY_HPB_Updated.npy")

        self.assertEqual(hpb_updated_by_santu.all(), hpb_updated_by_expectation_minimization_original.all())


if __name__ == '__main__':
    unittest.main()
