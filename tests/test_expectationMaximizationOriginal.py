# -*- coding: utf-8 -*-

from unittest import TestCase
from feature_mining import ExpectationMaximizationOriginal
import numpy as np


class TestExpectationMaximizationOriginal(TestCase):
    def test_e_step_hp(self):
        dump_path = "./tests/data/em_01/"

        em = ExpectationMaximizationOriginal(dump_path=dump_path)
        em.em()
        em._dump_hidden_parameters()

        hp_updated_by_santu = np.load(dump_path + "HP_updated.npy")
        hp_updated_by_expectation_minimization_original = np.load(dump_path + "MY_HP_Updated.npy")

        self.assertEqual(hp_updated_by_santu.all(), hp_updated_by_expectation_minimization_original.all())

    def test_e_step_hpb(self):
        dump_path = "./tests/data/em_01/"

        em = ExpectationMaximizationOriginal(dump_path=dump_path)
        em.em()
        em._dump_hidden_parameters()

        hpb_updated_by_santu = np.load(dump_path + "HPB_updated.npy")
        hpb_updated_by_expectation_minimization_original = np.load(dump_path + "MY_HPB_Updated.npy")

        self.assertEqual(hpb_updated_by_santu.all(), hpb_updated_by_expectation_minimization_original.all())

    def test_m_step(self):
        self.fail()
