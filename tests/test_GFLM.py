# -*- coding: utf-8 -*-

from unittest import TestCase
from feature_mining.gflm_tagger import GFLM
import numpy as np
import pandas as pd
from collections import defaultdict, OrderedDict, Counter
from scipy.sparse import csr_matrix


class TestGFLM(TestCase):

    def test_constructor(self):
        pass

    def test_small_results(self):
        pi_matrix = np.array([[.3, .7], [.6, .4]])
        hidden_background = np.array([[.8, .6, .1], [.1, .2, .3]])
        hidden_params = dict()
        hidden_params[0] = np.array([[.5, .4, .1], [.9, .8, .3]])
        hidden_params[1] = np.array([[.5, .6, .9], [.1, .2, .7]])

        gflm = GFLM(hidden_params=hidden_params, hidden_background=hidden_background, pi_matrix=pi_matrix)
        gflm.calc_gflm_section()
        gflm.calc_gflm_word()

        expected_gflm_word_all = pd.DataFrame([[0.16, 0, 0], [.81, 1, 0], [.81, 0, 1], [.49, 1, 1]],
                                              columns=["gflm_word", "section_id", "implicit_feature_id"])
        expected_gflm_word = pd.DataFrame([[.81, 1, 0], [.81, 0, 1], [.49, 1, 1]],
                                          columns=["gflm_word", "section_id", "implicit_feature_id"])

        expected_gflm_section_all = pd.DataFrame([[0.3, 0, 0], [.6, 1, 0], [.7, 0, 1], [.4, 1, 1]],
                                                 columns=["gflm_section", "section_id", "implicit_feature_id"])
        expected_gflm_section = pd.DataFrame([[.6, 1, 0], [.7, 0, 1], [.4, 1, 1]],
                                             columns=["gflm_section", "section_id", "implicit_feature_id"])

        self.assertEqual(True, pd.DataFrame.equals(expected_gflm_word, np.round(gflm.gflm_word, 2)),
                         msg="GFLM word mismatch")
        self.assertEqual(True, pd.DataFrame.equals(expected_gflm_word_all, np.round(gflm.gflm_word_all, 2)),
                         msg="GFLM word all mismatch")

        self.assertEqual(True, pd.DataFrame.equals(expected_gflm_section, gflm.gflm_section),
                         msg="GFLM section mismatch")
        self.assertEqual(True, pd.DataFrame.equals(expected_gflm_section_all, gflm.gflm_section_all),
                         msg="GFLM section all mismatch")


# added for exploratory testing
if __name__ == '__main__':
    pass
