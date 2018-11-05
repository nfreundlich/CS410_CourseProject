# -*- coding: utf-8 -*-

from .context import feature_mining
from feature_mining import EM
import numpy as np

import unittest


class AdvancedTestSuite(unittest.TestCase):
    """Advanced test cases."""

    def test_advanced_setup(self):
        import os
        print("Advanced test setup.")
        print("CWD:")
        print(os.getcwd())
        self.assertIsNone(None)

    def test_EM_Santu_E_Step_HP(self):
        dump_path = "./tests/data/em_01/"
        em = EM(dump_path = dump_path)
        em.em_e_step_dense()
        HP_Updated_by_Santu = np.load(dump_path + "HP_updated.npy")
        HPB_Updated_by_Santu = np.load(dump_path + "HPB_updated.npy")
        HP_Updated_by_E_Step_Dense = np.load(dump_path + "MY_HP_Updated.npy")
        HPB_Updated_by_E_Step_Dense = np.load(dump_path + "MY_HPB_Updated.npy")

        self.assertEqual(HP_Updated_by_Santu.all(), HP_Updated_by_E_Step_Dense.all())

    def test_EM_Santu_E_Step_HPB(self):
        dump_path = "./tests/data/em_01/"
        em = EM(dump_path = dump_path)
        em.em_e_step_dense()
        HPB_Updated_by_Santu = np.load(dump_path + "HPB_updated.npy")
        HPB_Updated_by_E_Step_Dense = np.load(dump_path + "MY_HPB_Updated.npy")

        self.assertEqual(HPB_Updated_by_Santu.all(), HPB_Updated_by_E_Step_Dense.all())


    def test_EM_E_Step_HPB(self):
        dump_path = "./tests/data/em_01/"
        em = EM(dump_path=dump_path)
        em.em_e_step_sparse()
        HPB_Updated_by_Santu = np.load(dump_path + "HPB_updated.npy")
        HPB_Updated_by_E_Step_Dense = np.load(dump_path + "MY_HPB_Updated.npy")

        self.assertEqual(HPB_Updated_by_Santu.all(), HPB_Updated_by_E_Step_Dense.all())

if __name__ == '__main__':
    unittest.main()
