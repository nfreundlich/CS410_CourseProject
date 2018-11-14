# -*- coding: utf-8 -*-

from unittest import TestCase
from feature_mining import ExpectationMaximizationOriginal
import numpy as np


class TestExpectationMaximizationOriginal(TestCase):
    """
    Tests for the original EM Algorithm as developed by Santu.
    """
    def __init__(self, *args, **kwargs):
        super(TestExpectationMaximizationOriginal, self).__init__(*args, **kwargs)
        #self.gen_stubs()

    def setUp(self):
        dump_path = "./tests/data/em_01/"

        em = ExpectationMaximizationOriginal(dump_path=dump_path)
        em.em()
        em._dump_hidden_parameters()

    def test_e_step_hp(self):
        dump_path = "./tests/data/em_01/"

        #em = ExpectationMaximizationOriginal(dump_path=dump_path)
        #em.em()
        #em._dump_hidden_parameters()

        hp_updated_by_santu = np.load(dump_path + "HP_updated.npy")
        hp_updated_by_expectation_minimization_original = np.load(dump_path + "MY_HP_Updated.npy")

        self.assertEqual(hp_updated_by_santu.all(), hp_updated_by_expectation_minimization_original.all())

    def test_e_step_hpb(self):
        dump_path = "./tests/data/em_01/"

        #em = ExpectationMaximizationOriginal(dump_path=dump_path)
        #em.em()
        #em._dump_hidden_parameters()

        hpb_updated_by_santu = np.load(dump_path + "HPB_updated.npy")
        hpb_updated_by_expectation_minimization_original = np.load(dump_path + "MY_HPB_Updated.npy")

        self.assertEqual(hpb_updated_by_santu.all(), hpb_updated_by_expectation_minimization_original.all())

    def test_m_step(self):
        dump_path = "./tests/data/em_01/"

        #em = ExpectationMaximizationOriginal(dump_path=dump_path)
        #em.em()
        #em._dump_hidden_parameters()

        pi_updated_by_santu = np.load(dump_path + "PI_updated.npy")
        pi_updated_by_expectation_minimization_original = np.load(dump_path + "MY_PI_Updated.npy")

        self.assertEqual(pi_updated_by_santu.all(), pi_updated_by_expectation_minimization_original.all())

    def test_previous_pi_copy(self):
        dump_path = "./tests/data/em_01/"
        #my_prev_pi = em.revious_pi

        pi_before = np.load(dump_path + "PI.npy")
        my_pi_before = np.load(dump_path + "MY_PREVIOUS_PI.npy")

        self.assertEqual(pi_before.all(), my_pi_before.all())

    def test_compute_cost(self):
        dump_path = "./tests/data/em_01/"

        cost = np.load(dump_path + "DIST.npy")
        my_cost = np.load(dump_path + "MY_DIST.npy")

        print(cost, my_cost)

        self.assertEqual(cost, my_cost)