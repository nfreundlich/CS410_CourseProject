# -*- coding: utf-8 -*-

from unittest import TestCase
from feature_mining.em_vector_by_feature import EmVectorByFeature
from feature_mining import ParseAndModel
import numpy as np
from scipy.sparse import csr_matrix
import os
import pickle
import string

def strip_punctuation(s):
    new_s = ''.join(c for c in s if c not in string.punctuation)
    return new_s.strip()

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


    def test_against_original_1_single_iteration(self):

        # load pi initalization
        infile = open("original_code_data/test_original_1_pi_init.data", 'rb')
        pi_init = pickle.load(infile)
        infile.close()

        # load pi params (end of iteration 1)
        infile = open("original_code_data/test_original_1_pi_params_it1.data", 'rb')
        pi_params_it1 = pickle.load(infile)
        infile.close()

        # load pi params (end of iteration 2)
        infile = open("original_code_data/test_original_1_pi_params_it2.data", 'rb')
        pi_params_it2 = pickle.load(infile)
        infile.close()

        # load hidden params (end of iteration 1)
        infile = open("original_code_data/test_original_1_hidden_params_it1.data", 'rb')
        hidden_params_it1 = pickle.load(infile)
        infile.close()

        # load hidden params (end of iteration 2)
        infile = open("original_code_data/test_original_1_hidden_params_it2.data", 'rb')
        hidden_params_it2 = pickle.load(infile)
        infile.close()

        # load hidden background params (end of iteration 1)
        infile = open("original_code_data/test_original_1_hidden_background_params_it1.data", 'rb')
        hidden_back_params_it1 = pickle.load(infile)
        infile.close()

        # load hidden background params (end of iteration 2)
        infile = open("original_code_data/test_original_1_hidden_background_params_it2.data", 'rb')
        hidden_back_params_it2 = pickle.load(infile)
        infile.close()

        pm_inst = ParseAndModel(feature_list=["sound", "battery"], filename='data/parse_and_model/iPod.final', lemmatize_words=False, log_base=None, start_line=4, nlines=11, include_title_lines=False)

        em = EmVectorByFeature(explicit_model=pm_inst)
        em.initialize_parameters()

        # use fixed init
        # check section word counts
        for review_id in range(0, len(pi_init)):
            review_sections = pm_inst.parsed_text["section_list"]
            review_sections = review_sections[review_sections.doc_id == review_id]
            section_og_id = 0
            for section_index, section_row in review_sections.iterrows():
                for feature_id in range(0, len(pi_init[review_id][section_og_id])):
                    feature_name_row = pm_inst.formatted_feature_list[pm_inst.formatted_feature_list.feature_id == feature_id]
                    feature_name_row = feature_name_row.reset_index(drop=True)
                    em.pi_matrix[section_index,feature_id] = pi_init[review_id][section_og_id][feature_name_row["feature"][0]]

                section_og_id += 1

        # set iterations
        em.max_iter=1
        em.lambda_background= 0.7

        em.em_loop()

        # Check hidden parameters
        dense_hidden_params = list()
        for feature_id in range(0, len(em.hidden_parameters)):
            dense_hidden_params.append(em.hidden_parameters[feature_id].toarray())
        inverse_vocab_lookup = {strip_punctuation(v): k for k, v in pm_inst.model_results["vocabulary_lookup"].items()}

        for review_id in range(0, len(hidden_params_it1)):
            review_sections = pm_inst.parsed_text["section_list"]
            review_sections = review_sections[review_sections.doc_id == review_id]
            section_og_id = 0
            for section_index, section_row in review_sections.iterrows():
                for word in hidden_params_it1[review_id][section_og_id].keys():
                    word_id = inverse_vocab_lookup[strip_punctuation(word)]
                    for feature_id in range(0, len(hidden_params_it1[review_id][section_og_id][word])):
                        feature_name_row = pm_inst.formatted_feature_list[pm_inst.formatted_feature_list.feature_id == feature_id]
                        feature_name_row = feature_name_row.reset_index(drop=True)
                        actual_param = dense_hidden_params[feature_id][section_row["section_id"], word_id]
                        original_param = hidden_params_it1[review_id][section_og_id][word][feature_name_row["feature"][0]]

                        print("checking word:" + word)
                        self.assertEqual(round(actual_param, 8), round(original_param, 8),
                                         msg="hidden feature - feature_id: " + str(
                                             feature_name_row["feature"][0]) + ", word=" + word + ", a=" + str(
                                             actual_param) + ", e=" + str(original_param))

                section_og_id += 1

        # Check hidden background parameters
        dense_hidden_back_params = em.hidden_parameters_background.toarray()
        for review_id in range(0, len(hidden_back_params_it1)):
            review_sections = pm_inst.parsed_text["section_list"]
            review_sections = review_sections[review_sections.doc_id == review_id]
            section_og_id = 0
            for section_index, section_row in review_sections.iterrows():
                for word in hidden_params_it1[review_id][section_og_id].keys():
                    word_id = inverse_vocab_lookup[strip_punctuation(word)]
                    actual_param = dense_hidden_back_params[section_row["section_id"], word_id]
                    original_param = hidden_back_params_it1[review_id][section_og_id][word]

                    print("checking word:" + word)
                    self.assertEqual(round(actual_param, 8), round(original_param, 8),
                                     msg="hidden background: " + ", word=" + word + ", a=" + str(
                                         actual_param) + ", e=" + str(original_param))

                section_og_id += 1

        # Check pi parameters
        for review_id in range(0, len(pi_params_it1)):
            review_sections = pm_inst.parsed_text["section_list"]
            review_sections = review_sections[review_sections.doc_id == review_id]
            section_og_id = 0
            for section_index, section_row in review_sections.iterrows():
                for feature_id in range(0, len(pi_params_it1[review_id][section_og_id])):
                    feature_name_row = pm_inst.formatted_feature_list[pm_inst.formatted_feature_list.feature_id == feature_id]
                    feature_name_row = feature_name_row.reset_index(drop=True)
                    actual_param = em.pi_matrix[section_index,feature_id]
                    original_param = pi_params_it1[review_id][section_og_id][feature_name_row["feature"][0]]

                    print("checking section:" + str(section_index))
                    self.assertEqual(round(actual_param, 8), round(original_param, 8),
                                     msg="pi params: " + ", feature=" + str(feature_name_row["feature"][0]) +
                                         ", section= " + str(section_index) + ", a=" + str(
                                         actual_param) + ", e=" + str(original_param))

                section_og_id += 1


        # check section word counts
        pm_section_word_counts = pm.model_results["section_word_counts_matrix"].toarray()
        inverse_vocab_lookup = {strip_punctuation(v): k for k, v in pm.model_results["vocabulary_lookup"].items()}
        for review_id in range(0, len(section_word_counts)):
            print("SWC - Checking review: " + str(review_id))
            review_sections = pm.parsed_text["section_list"]
            review_sections = review_sections[review_sections.doc_id == review_id]
            section_og_id = 0
            for section_index, section_row in review_sections.iterrows():
                print("SWC - Checking section:" +str(section_row["section_id"]))
                for word in section_word_counts[review_id][section_og_id].keys():
                    word = strip_punctuation(word)
                    if word == '':
                        continue
                    vocab_word_id = inverse_vocab_lookup[word]
                    actual_count = pm_section_word_counts[section_row["section_id"], vocab_word_id]
                    original_count = section_word_counts[review_id][section_og_id][word]

                    self.assertEqual(actual_count, original_count, msg="SWC - section_id: " + str(section_row["section_id"]) + ", " + word + " a=" + str(actual_count) + ", e=" + str(original_count))

                section_og_id+=1

        # check background model
        for word in background_model.keys():
            print("Background - Checking word:" + word)
            word = strip_punctuation(word)
            vocab_word_id = inverse_vocab_lookup[word]

            actual_prob = pm.model_results["model_background"][vocab_word_id]
            original_prob = background_model[word]

            self.assertEqual(actual_prob, original_prob,
                             msg="Background prob:" + word + " a=" + str(
                                 actual_prob) + ", e=" + str(original_prob))


        # check topic model
        for f_index, feature_row in pm.formatted_feature_list.iterrows():
            for word_index, word in pm.model_results["vocabulary_lookup"].items():
                print("Topic - Checking word:" + word)

                word = strip_punctuation(word)
                feature_index = feature_row["feature_id"]
                actual_prob = pm.model_results["model_feature"][feature_index][word_index]
                original_prob = topic_model[feature_row["feature"]][word]

                self.assertEqual(round(actual_prob,8), round(original_prob,8),
                                 msg="topic - feature_id: " + str(feature_row["feature_id"]) + ", word=" + word + ", a=" + str(
                                     actual_prob) + ", e=" + str(original_prob))

    def test_compute_denom(self):

        self.assertEqual(True, 1 < 0.001)

    def test_compute_nom(self):

        self.assertEqual(True, 1 < 0.001)

    def test_m_step(self):

        self.assertEqual(True, False)

