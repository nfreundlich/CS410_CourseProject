# -*- coding: utf-8 -*-

from unittest import TestCase
from feature_mining.em_vector_by_feature import EmVectorByFeature
import os


class TestEmVectorByFeature(TestCase):
    def __init__(self, *args, **kwargs):
        super(TestEmVectorByFeature, self).__init__(*args, **kwargs)

    def setUp(self):
        """
        Special test method: setUp
        Will be executed before tests.
        :return: none
        """
        print("CWD: ", os.getcwd())
        if 'tests' in os.getcwd():
            self.dump_path = "./data/em_01/"
        else:
            self.dump_path = "./tests/data/em_01/"

        em = EmVectorByFeature()

    def tearDown(self):
        """
        Special test method: tearDown()
        Used to clean up tests and configuration, if needed, after test execution.
        :return: none
        """

    def test_e_step(self):

        self.assertEqual(True, False)

    def test_compute_denom(self):

        self.assertEqual(True, 1 < 0.001)

    def test_compute_nom(self):

        self.assertEqual(True, 1 < 0.001)

    def test_m_step(self):


        self.assertEqual(True, False)

