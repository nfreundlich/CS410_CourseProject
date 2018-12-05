# -*- coding: utf-8 -*-
import unittest
from unittest import TestCase
from feature_mining import ExpectationMaximizationVector
import numpy as np
import os


class TestExpectationMaximizationVector(TestCase):
    def __init__(self, *args, **kwargs):
        super(TestExpectationMaximizationVector, self).__init__(*args, **kwargs)

    def setUp(self):
        print("CWD: ", os.getcwd())
        if 'tests' in os.getcwd():
            self.dump_path = "./data/em_01/"
        else:
            self.dump_path = "./tests/data/em_01/"


    def test_e_step_hp_01(self):
        em = ExpectationMaximizationVector(dump_path=self.dump_path)
        em.em()

        hp_em_vector_one_sentence_for_testing = em.hidden_parameters[0]

        print("Values computed by e_step_vector:")
        hp_updated_by_santu = np.load(self.dump_path + "HP_updated.npy")
        sentence = 0
        print(em.features_map.keys())
        for i in np.where(em.reviews_matrix[sentence].todense() > 0)[1]:
            print(em.words_list[i], hp_em_vector_one_sentence_for_testing[i])
        print("Values computed by e_step_original")
        for key in hp_updated_by_santu[0][0]:
            print(key, hp_updated_by_santu[0][0][key])

        first_step_ok = True
        features_list = []
        for k, v in em.features_map.items():
            features_list.append(k)
        for i in np.where(em.reviews_matrix[sentence].todense() > 0)[1]:
            print(em.words_list[i])
            for j in range(0, len(np.array(hp_em_vector_one_sentence_for_testing[i]).squeeze())):
                print(features_list[j], np.array(hp_em_vector_one_sentence_for_testing[i]).squeeze()[j])
                print(hp_updated_by_santu[0][0][em.words_list[i]][features_list[j]])
                if np.array(hp_em_vector_one_sentence_for_testing[i]).squeeze()[j] - \
                        hp_updated_by_santu[0][0][em.words_list[i]][features_list[j]] > 0.001:
                    first_step_ok = False
                    break
        self.assertEqual(True, first_step_ok)

    def test_e_step_hpb_01(self):
        em = ExpectationMaximizationVector(dump_path=self.dump_path)
        em.em()

        hp_background_em_vector = em.hidden_parameters_background[0].todense().T
        hpb_updated_by_expectation_minimization_original = np.load(self.dump_path + "HPB_Updated.npy")

        background_one_sentence_ok = True
        for i in np.where(hp_background_em_vector > 0)[0]:
            print(em.words_list[i], hp_background_em_vector[i].item())
            print(hpb_updated_by_expectation_minimization_original[0][0][em.words_list[i]])
            if hp_background_em_vector[i].item() - hpb_updated_by_expectation_minimization_original[0][0][em.words_list[i]] > 0.001:
                background_one_sentence_ok = False

        self.assertEqual(True, background_one_sentence_ok)

    @unittest.skip("DENOM not accessible. To reactivate when file recovered.")
    def test_compute_denom(self):
        em = ExpectationMaximizationVector(dump_path=self.dump_path)
        em.em()

        my_denom = em.denom
        denom_original = np.load(self.dump_path + "DENOM.npy").item()

        print("denom original: ", denom_original)
        print("my denom: " , my_denom)
        print("my denom detail: ", em.m_sum)

        self.assertEqual(True, np.fabs(my_denom - denom_original) < 0.001)

    @unittest.skip("NOM not accessible. To reactivate when file recovered.")
    def test_compute_nom(self):
        em = ExpectationMaximizationVector(dump_path=self.dump_path)
        em.em()

        my_nom = em.nom
        nom_original = np.load(self.dump_path + "NOM.npy").item()

        print("nom original: ", nom_original)
        print("my nom: " , my_nom)

        self.assertEqual(True, np.fabs(my_nom - nom_original) < 0.001)

    def test_m_step(self):
        em = ExpectationMaximizationVector(dump_path=self.dump_path)
        em.em()

        pi_updated_by_santu = np.load(self.dump_path + "PI_updated.npy")
        pi_updated = em.pi_matrix[0].reshape(1,em.f)

        print("Pi 0 updated by original", pi_updated_by_santu)
        print("Pi 0 updated by vector", pi_updated)

        is_ok = True
        for k, v in pi_updated_by_santu[0][0].items():
            print(k, v, pi_updated.item(0, em.features_map[k]))
            if np.fabs(v - pi_updated.item(0, em.features_map[k])) > 0.001:
                is_ok = False

        self.assertEqual(True, is_ok)

