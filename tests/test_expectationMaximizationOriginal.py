# -*- coding: utf-8 -*-

from unittest import TestCase
from feature_mining import ExpectationMaximizationOriginal
import numpy as np
import os


class TestExpectationMaximizationOriginal(TestCase):
    """
    Tests for the original EM Algorithm as developed by Santu.
    """
    def __init__(self, *args, **kwargs):
        super(TestExpectationMaximizationOriginal, self).__init__(*args, **kwargs)

    def setUp(self):
        print("Current working directory: ", os.getcwd())
        self.dump_path = "./data/em_01/"
        self.em = ExpectationMaximizationOriginal(dump_path=self.dump_path)
        self.em.em()
        self.em._dump_hidden_parameters()


    def test_e_step_hp(self):
        hp_updated_by_santu = np.load(self.dump_path + "HP_updated.npy")
        hp_updated_by_expectation_minimization_original = np.load(self.dump_path + "MY_HP_Updated.npy")

        self.assertEqual(hp_updated_by_santu.all(), hp_updated_by_expectation_minimization_original.all())

    def test_e_step_hpb(self):
        hpb_updated_by_santu = np.load(self.dump_path + "HPB_updated.npy")
        hpb_updated_by_expectation_minimization_original = np.load(self.dump_path + "MY_HPB_Updated.npy")

        self.assertEqual(hpb_updated_by_santu.all(), hpb_updated_by_expectation_minimization_original.all())

    def test_m_step(self):
        pi_updated_by_santu = np.load(self.dump_path + "PI_updated.npy")
        pi_updated_by_expectation_minimization_original = np.load(self.dump_path + "MY_PI_Updated.npy")

        self.assertEqual(pi_updated_by_santu.all(), pi_updated_by_expectation_minimization_original.all())

    def test_previous_pi_copy(self):
        pi_before = np.load(self.dump_path + "PI.npy")
        my_pi_before = np.load(self.dump_path + "MY_PREVIOUS_PI.npy")

        self.assertEqual(pi_before.all(), my_pi_before.all())

    def test_compute_cost(self):
        cost = np.load(self.dump_path + "DIST.npy")
        my_cost = np.load(self.dump_path + "MY_DIST.npy")
        print(cost, my_cost)

        self.assertEqual(cost, my_cost)
