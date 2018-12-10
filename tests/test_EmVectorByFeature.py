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
                                filename='../tests/data/parse_and_model/twoLineTest.txt', log_base=2)
        em = EmVectorByFeature(explicit_model=pm_inst)

        expected_section_word_counts_matrix = [[1, 1, 1, 0, 0], [1, 0, 0, 1, 1]]
        expected_model_background_matrix = np.array([1 / 3, 1 / 6, 1 / 6, 1 / 6, 1 / 6])
        expected_model_feature_matrix = np.array([[0.218], [0.282], [0.282], [0.109], [0.109]])

        self.assertEqual(True,
                         np.array_equiv(expected_section_word_counts_matrix,
                                        csr_matrix.toarray(em.reviews_matrix)), msg="section counts do not match")
        self.assertEqual(True, np.array_equiv(expected_model_background_matrix,
                                              csr_matrix.toarray(em.background_probability)),
                         msg="background model does not match")
        self.assertEqual(True, np.array_equiv(expected_model_feature_matrix,
                                              np.round(em.topic_model, 3)), msg="topic models do not match")

        print("testing")

    # TODO: if time later, abstract comparison code so it doesn't need to be repeatedly copied and pasted
    def test_against_original_1_single_iteration(self):

        # load pi initalization
        infile = open("original_code_data/test_original_1_pi_init.data", 'rb')
        pi_init = pickle.load(infile)
        infile.close()

        # load pi params (end of iteration 1)
        infile = open("original_code_data/test_original_1_pi_params_it1.data", 'rb')
        pi_params = pickle.load(infile)
        infile.close()

        # load hidden params (end of iteration 1)
        infile = open("original_code_data/test_original_1_hidden_params_it1.data", 'rb')
        hidden_params = pickle.load(infile)
        infile.close()

        # load hidden background params (end of iteration 1)
        infile = open("original_code_data/test_original_1_hidden_background_params_it1.data", 'rb')
        hidden_back_params = pickle.load(infile)
        infile.close()

        # load pi deltas
        infile = open("original_code_data/test_original_1_pi_delta_it1.data", 'rb')
        pi_delta = pickle.load(infile)
        infile.close()

        pm_inst = ParseAndModel(feature_list=["sound", "battery"], filename='data/parse_and_model/iPod.final',
                                lemmatize_words=False, log_base=None, start_line=4, nlines=11,
                                include_title_lines=False)

        # use fixed init
        # check section word counts
        pi_init_em = np.empty([pm_inst.model_results["section_word_counts_matrix"].shape[0], len(pm_inst.model_results["model_feature"])])
        for review_id in range(0, len(pi_init)):
            review_sections = pm_inst.parsed_text["section_list"]
            review_sections = review_sections[review_sections.doc_id == review_id]
            section_og_id = 0
            for section_index, section_row in review_sections.iterrows():
                for feature_id in range(0, len(pi_init[review_id][section_og_id])):
                    feature_name_row = pm_inst.formatted_feature_list[
                        pm_inst.formatted_feature_list.feature_id == feature_id]
                    feature_name_row = feature_name_row.reset_index(drop=True)
                    pi_init_em[section_index, feature_id] = pi_init[review_id][section_og_id][
                        feature_name_row["feature"][0]]

                section_og_id += 1

        em = EmVectorByFeature(explicit_model=pm_inst, max_iter=1, lambda_background=0.7, pi_init=pi_init_em)

        # TODO: this should probably be moved into a constructor somewhere
        em.initialize_parameters()

        em.em_loop()

        # Check hidden parameters
        dense_hidden_params = list()
        for feature_id in range(0, len(em.hidden_parameters)):
            dense_hidden_params.append(em.hidden_parameters[feature_id].toarray())
        inverse_vocab_lookup = {strip_punctuation(v): k for k, v in pm_inst.model_results["vocabulary_lookup"].items()}

        for review_id in range(0, len(hidden_params)):
            review_sections = pm_inst.parsed_text["section_list"]
            review_sections = review_sections[review_sections.doc_id == review_id]
            section_og_id = 0
            for section_index, section_row in review_sections.iterrows():
                for word in hidden_params[review_id][section_og_id].keys():
                    word_id = inverse_vocab_lookup[strip_punctuation(word)]
                    for feature_id in range(0, len(hidden_params[review_id][section_og_id][word])):
                        feature_name_row = pm_inst.formatted_feature_list[
                            pm_inst.formatted_feature_list.feature_id == feature_id]
                        feature_name_row = feature_name_row.reset_index(drop=True)
                        actual_param = dense_hidden_params[feature_id][section_row["section_id"], word_id]
                        original_param = hidden_params[review_id][section_og_id][word][feature_name_row["feature"][0]]

                        print("checking word:" + word)
                        self.assertEqual(round(actual_param, 8), round(original_param, 8),
                                         msg="hidden feature - feature_id: " + str(
                                             feature_name_row["feature"][0]) + ", word=" + word + ", a=" + str(
                                             actual_param) + ", e=" + str(original_param))

                section_og_id += 1

        # Check hidden background parameters
        dense_hidden_back_params = em.hidden_parameters_background.toarray()
        for review_id in range(0, len(hidden_back_params)):
            review_sections = pm_inst.parsed_text["section_list"]
            review_sections = review_sections[review_sections.doc_id == review_id]
            section_og_id = 0
            for section_index, section_row in review_sections.iterrows():
                for word in hidden_params[review_id][section_og_id].keys():
                    word_id = inverse_vocab_lookup[strip_punctuation(word)]
                    actual_param = dense_hidden_back_params[section_row["section_id"], word_id]
                    original_param = hidden_back_params[review_id][section_og_id][word]

                    print("checking word:" + word)
                    self.assertEqual(round(actual_param, 8), round(original_param, 8),
                                     msg="hidden background: " + ", word=" + word + ", a=" + str(
                                         actual_param) + ", e=" + str(original_param))

                section_og_id += 1

        # Check pi parameters
        for review_id in range(0, len(pi_params)):
            review_sections = pm_inst.parsed_text["section_list"]
            review_sections = review_sections[review_sections.doc_id == review_id]
            section_og_id = 0
            for section_index, section_row in review_sections.iterrows():
                for feature_id in range(0, len(pi_params[review_id][section_og_id])):
                    feature_name_row = pm_inst.formatted_feature_list[
                        pm_inst.formatted_feature_list.feature_id == feature_id]
                    feature_name_row = feature_name_row.reset_index(drop=True)
                    actual_param = em.pi_matrix[section_index, feature_id]
                    original_param = pi_params[review_id][section_og_id][feature_name_row["feature"][0]]

                    print("checking section:" + str(section_index))
                    self.assertEqual(round(actual_param, 8), round(original_param, 8),
                                     msg="pi params: " + ", feature=" + str(feature_name_row["feature"][0]) +
                                         ", section= " + str(section_index) + ", a=" + str(
                                         actual_param) + ", e=" + str(original_param))

                section_og_id += 1

        # Check pi deltas
        self.assertEqual(round(pi_delta, 8), round(em.pi_delta, 8),
                         msg="pi delta: " + ", a=" + str(
                             em.pi_delta) + ", e=" + str(pi_delta))

    def test_against_original_1_double_iteration(self):

        # load pi initalization
        infile = open("original_code_data/test_original_1_pi_init.data", 'rb')
        pi_init = pickle.load(infile)
        infile.close()

        # load pi params (end of iteration 2)
        infile = open("original_code_data/test_original_1_pi_params_it2.data", 'rb')
        pi_params = pickle.load(infile)
        infile.close()

        # load hidden params (end of iteration 2)
        infile = open("original_code_data/test_original_1_hidden_params_it2.data", 'rb')
        hidden_params = pickle.load(infile)
        infile.close()

        # load hidden background params (end of iteration 2)
        infile = open("original_code_data/test_original_1_hidden_background_params_it2.data", 'rb')
        hidden_back_params = pickle.load(infile)
        infile.close()

        # load pi deltas
        infile = open("original_code_data/test_original_1_pi_delta_it2.data", 'rb')
        pi_delta = pickle.load(infile)
        infile.close()

        pm_inst = ParseAndModel(feature_list=["sound", "battery"], filename='data/parse_and_model/iPod.final',
                                lemmatize_words=False, log_base=None, start_line=4, nlines=11,
                                include_title_lines=False)

        # use fixed init
        # check section word counts
        pi_init_em = np.empty(
            [pm_inst.model_results["section_word_counts_matrix"].shape[0], len(pm_inst.model_results["model_feature"])])
        for review_id in range(0, len(pi_init)):
            review_sections = pm_inst.parsed_text["section_list"]
            review_sections = review_sections[review_sections.doc_id == review_id]
            section_og_id = 0
            for section_index, section_row in review_sections.iterrows():
                for feature_id in range(0, len(pi_init[review_id][section_og_id])):
                    feature_name_row = pm_inst.formatted_feature_list[
                        pm_inst.formatted_feature_list.feature_id == feature_id]
                    feature_name_row = feature_name_row.reset_index(drop=True)
                    pi_init_em[section_index, feature_id] = pi_init[review_id][section_og_id][
                        feature_name_row["feature"][0]]

                section_og_id += 1

        em = EmVectorByFeature(explicit_model=pm_inst, max_iter=2, lambda_background=0.7, pi_init=pi_init_em)

        # TODO: this should probably be moved into a constructor somewhere
        em.initialize_parameters()

        em.em_loop()

        # Check hidden parameters
        dense_hidden_params = list()
        for feature_id in range(0, len(em.hidden_parameters)):
            dense_hidden_params.append(em.hidden_parameters[feature_id].toarray())
        inverse_vocab_lookup = {strip_punctuation(v): k for k, v in pm_inst.model_results["vocabulary_lookup"].items()}

        for review_id in range(0, len(hidden_params)):
            review_sections = pm_inst.parsed_text["section_list"]
            review_sections = review_sections[review_sections.doc_id == review_id]
            section_og_id = 0
            for section_index, section_row in review_sections.iterrows():
                for word in hidden_params[review_id][section_og_id].keys():
                    word_id = inverse_vocab_lookup[strip_punctuation(word)]
                    for feature_id in range(0, len(hidden_params[review_id][section_og_id][word])):
                        feature_name_row = pm_inst.formatted_feature_list[
                            pm_inst.formatted_feature_list.feature_id == feature_id]
                        feature_name_row = feature_name_row.reset_index(drop=True)
                        actual_param = dense_hidden_params[feature_id][section_row["section_id"], word_id]
                        original_param = hidden_params[review_id][section_og_id][word][feature_name_row["feature"][0]]

                        print("checking word:" + word)
                        self.assertEqual(round(actual_param, 8), round(original_param, 8),
                                         msg="hidden feature - feature_id: " + str(
                                             feature_name_row["feature"][0]) + ", word=" + word + ", a=" + str(
                                             actual_param) + ", e=" + str(original_param))

                section_og_id += 1

        # Check hidden background parameters
        dense_hidden_back_params = em.hidden_parameters_background.toarray()
        for review_id in range(0, len(hidden_back_params)):
            review_sections = pm_inst.parsed_text["section_list"]
            review_sections = review_sections[review_sections.doc_id == review_id]
            section_og_id = 0
            for section_index, section_row in review_sections.iterrows():
                for word in hidden_params[review_id][section_og_id].keys():
                    word_id = inverse_vocab_lookup[strip_punctuation(word)]
                    actual_param = dense_hidden_back_params[section_row["section_id"], word_id]
                    original_param = hidden_back_params[review_id][section_og_id][word]

                    print("checking word:" + word)
                    self.assertEqual(round(actual_param, 8), round(original_param, 8),
                                     msg="hidden background: " + ", word=" + word + ", a=" + str(
                                         actual_param) + ", e=" + str(original_param))

                section_og_id += 1

        # Check pi parameters
        for review_id in range(0, len(pi_params)):
            review_sections = pm_inst.parsed_text["section_list"]
            review_sections = review_sections[review_sections.doc_id == review_id]
            section_og_id = 0
            for section_index, section_row in review_sections.iterrows():
                for feature_id in range(0, len(pi_params[review_id][section_og_id])):
                    feature_name_row = pm_inst.formatted_feature_list[
                        pm_inst.formatted_feature_list.feature_id == feature_id]
                    feature_name_row = feature_name_row.reset_index(drop=True)
                    actual_param = em.pi_matrix[section_index, feature_id]
                    original_param = pi_params[review_id][section_og_id][feature_name_row["feature"][0]]

                    print("checking section:" + str(section_index))
                    self.assertEqual(round(actual_param, 8), round(original_param, 8),
                                     msg="pi params: " + ", feature=" + str(feature_name_row["feature"][0]) +
                                         ", section= " + str(section_index) + ", a=" + str(
                                         actual_param) + ", e=" + str(original_param))

                section_og_id += 1

        # Check pi deltas
        self.assertEqual(round(pi_delta, 8), round(em.pi_delta, 8),
                         msg="pi delta: " + ", a=" + str(
                             em.pi_delta) + ", e=" + str(pi_delta))

        def test_against_original_1_double_iteration(self):

            # load pi initalization
            infile = open("original_code_data/test_original_1_pi_init.data", 'rb')
            pi_init = pickle.load(infile)
            infile.close()

            # load pi params (end of iteration 2)
            infile = open("original_code_data/test_original_1_pi_params_it2.data", 'rb')
            pi_params = pickle.load(infile)
            infile.close()

            # load hidden params (end of iteration 2)
            infile = open("original_code_data/test_original_1_hidden_params_it2.data", 'rb')
            hidden_params = pickle.load(infile)
            infile.close()

            # load hidden background params (end of iteration 2)
            infile = open("original_code_data/test_original_1_hidden_background_params_it2.data", 'rb')
            hidden_back_params = pickle.load(infile)
            infile.close()

            # load pi deltas
            infile = open("original_code_data/test_original_1_pi_delta_it2.data", 'rb')
            pi_delta = pickle.load(infile)
            infile.close()

            pm_inst = ParseAndModel(feature_list=["sound", "battery"], filename='data/parse_and_model/iPod.final',
                                    lemmatize_words=False, log_base=None, start_line=4, nlines=11,
                                    include_title_lines=False)

            # use fixed init
            # check section word counts
            pi_init_em = np.empty([pm_inst.model_results["section_word_counts_matrix"].shape[0],
                                   len(pm_inst.model_results["model_feature"])])
            for review_id in range(0, len(pi_init)):
                review_sections = pm_inst.parsed_text["section_list"]
                review_sections = review_sections[review_sections.doc_id == review_id]
                section_og_id = 0
                for section_index, section_row in review_sections.iterrows():
                    for feature_id in range(0, len(pi_init[review_id][section_og_id])):
                        feature_name_row = pm_inst.formatted_feature_list[
                            pm_inst.formatted_feature_list.feature_id == feature_id]
                        feature_name_row = feature_name_row.reset_index(drop=True)
                        pi_init_em[section_index, feature_id] = pi_init[review_id][section_og_id][
                            feature_name_row["feature"][0]]

                    section_og_id += 1

            em = EmVectorByFeature(explicit_model=pm_inst, max_iter=50, lambda_background=0.7, pi_init=pi_init_em)

            # TODO: this should probably be moved into a constructor somewhere
            em.initialize_parameters()

            em.em_loop()

            # Check hidden parameters
            dense_hidden_params = list()
            for feature_id in range(0, len(em.hidden_parameters)):
                dense_hidden_params.append(em.hidden_parameters[feature_id].toarray())
            inverse_vocab_lookup = {strip_punctuation(v): k for k, v in
                                    pm_inst.model_results["vocabulary_lookup"].items()}

            for review_id in range(0, len(hidden_params)):
                review_sections = pm_inst.parsed_text["section_list"]
                review_sections = review_sections[review_sections.doc_id == review_id]
                section_og_id = 0
                for section_index, section_row in review_sections.iterrows():
                    for word in hidden_params[review_id][section_og_id].keys():
                        word_id = inverse_vocab_lookup[strip_punctuation(word)]
                        for feature_id in range(0, len(hidden_params[review_id][section_og_id][word])):
                            feature_name_row = pm_inst.formatted_feature_list[
                                pm_inst.formatted_feature_list.feature_id == feature_id]
                            feature_name_row = feature_name_row.reset_index(drop=True)
                            actual_param = dense_hidden_params[feature_id][section_row["section_id"], word_id]
                            original_param = hidden_params[review_id][section_og_id][word][
                                feature_name_row["feature"][0]]

                            print("checking word:" + word)
                            self.assertEqual(round(actual_param, 8), round(original_param, 8),
                                             msg="hidden feature - feature_id: " + str(
                                                 feature_name_row["feature"][0]) + ", word=" + word + ", a=" + str(
                                                 actual_param) + ", e=" + str(original_param))

                    section_og_id += 1

            # Check hidden background parameters
            dense_hidden_back_params = em.hidden_parameters_background.toarray()
            for review_id in range(0, len(hidden_back_params)):
                review_sections = pm_inst.parsed_text["section_list"]
                review_sections = review_sections[review_sections.doc_id == review_id]
                section_og_id = 0
                for section_index, section_row in review_sections.iterrows():
                    for word in hidden_params[review_id][section_og_id].keys():
                        word_id = inverse_vocab_lookup[strip_punctuation(word)]
                        actual_param = dense_hidden_back_params[section_row["section_id"], word_id]
                        original_param = hidden_back_params[review_id][section_og_id][word]

                        print("checking word:" + word)
                        self.assertEqual(round(actual_param, 8), round(original_param, 8),
                                         msg="hidden background: " + ", word=" + word + ", a=" + str(
                                             actual_param) + ", e=" + str(original_param))

                    section_og_id += 1

            # Check pi parameters
            for review_id in range(0, len(pi_params)):
                review_sections = pm_inst.parsed_text["section_list"]
                review_sections = review_sections[review_sections.doc_id == review_id]
                section_og_id = 0
                for section_index, section_row in review_sections.iterrows():
                    for feature_id in range(0, len(pi_params[review_id][section_og_id])):
                        feature_name_row = pm_inst.formatted_feature_list[
                            pm_inst.formatted_feature_list.feature_id == feature_id]
                        feature_name_row = feature_name_row.reset_index(drop=True)
                        actual_param = em.pi_matrix[section_index, feature_id]
                        original_param = pi_params[review_id][section_og_id][feature_name_row["feature"][0]]

                        print("checking section:" + str(section_index))
                        self.assertEqual(round(actual_param, 8), round(original_param, 8),
                                         msg="pi params: " + ", feature=" + str(feature_name_row["feature"][0]) +
                                             ", section= " + str(section_index) + ", a=" + str(
                                             actual_param) + ", e=" + str(original_param))

                    section_og_id += 1

            # Check pi deltas
            self.assertEqual(round(pi_delta, 8), round(em.pi_delta, 8),
                             msg="pi delta: " + ", a=" + str(
                                 em.pi_delta) + ", e=" + str(pi_delta))

    def test_against_original_1_50_iteration(self):

        # load pi initalization
        infile = open("original_code_data/test_original_1_pi_init.data", 'rb')
        pi_init = pickle.load(infile)
        infile.close()

        # load pi params (end of iteration 2)
        infile = open("original_code_data/test_original_1_pi_params_it50.data", 'rb')
        pi_params = pickle.load(infile)
        infile.close()

        # load hidden params (end of iteration 2)
        infile = open("original_code_data/test_original_1_hidden_params_it50.data", 'rb')
        hidden_params = pickle.load(infile)
        infile.close()

        # load hidden background params (end of iteration 2)
        infile = open("original_code_data/test_original_1_hidden_background_params_it50.data", 'rb')
        hidden_back_params = pickle.load(infile)
        infile.close()

        # load pi deltas
        infile = open("original_code_data/test_original_1_pi_delta_it50.data", 'rb')
        pi_delta = pickle.load(infile)
        infile.close()

        pm_inst = ParseAndModel(feature_list=["sound", "battery"], filename='data/parse_and_model/iPod.final',
                                lemmatize_words=False, log_base=None, start_line=4, nlines=11,
                                include_title_lines=False)

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
                    feature_name_row = pm_inst.formatted_feature_list[
                        pm_inst.formatted_feature_list.feature_id == feature_id]
                    feature_name_row = feature_name_row.reset_index(drop=True)
                    em.pi_matrix[section_index, feature_id] = pi_init[review_id][section_og_id][
                        feature_name_row["feature"][0]]

                section_og_id += 1

        # set iterations
        em.max_iter = 50
        em.lambda_background = 0.7

        em.em_loop()

        # Check hidden parameters
        dense_hidden_params = list()
        for feature_id in range(0, len(em.hidden_parameters)):
            dense_hidden_params.append(em.hidden_parameters[feature_id].toarray())
        inverse_vocab_lookup = {strip_punctuation(v): k for k, v in pm_inst.model_results["vocabulary_lookup"].items()}

        for review_id in range(0, len(hidden_params)):
            review_sections = pm_inst.parsed_text["section_list"]
            review_sections = review_sections[review_sections.doc_id == review_id]
            section_og_id = 0
            for section_index, section_row in review_sections.iterrows():
                for word in hidden_params[review_id][section_og_id].keys():
                    word_id = inverse_vocab_lookup[strip_punctuation(word)]
                    for feature_id in range(0, len(hidden_params[review_id][section_og_id][word])):
                        feature_name_row = pm_inst.formatted_feature_list[
                            pm_inst.formatted_feature_list.feature_id == feature_id]
                        feature_name_row = feature_name_row.reset_index(drop=True)
                        actual_param = dense_hidden_params[feature_id][section_row["section_id"], word_id]
                        original_param = hidden_params[review_id][section_og_id][word][feature_name_row["feature"][0]]

                        print("checking word:" + word)
                        self.assertEqual(round(actual_param, 8), round(original_param, 8),
                                         msg="hidden feature - feature_id: " + str(
                                             feature_name_row["feature"][0]) + ", word=" + word + ", a=" + str(
                                             actual_param) + ", e=" + str(original_param))

                section_og_id += 1

        # Check hidden background parameters
        dense_hidden_back_params = em.hidden_parameters_background.toarray()
        for review_id in range(0, len(hidden_back_params)):
            review_sections = pm_inst.parsed_text["section_list"]
            review_sections = review_sections[review_sections.doc_id == review_id]
            section_og_id = 0
            for section_index, section_row in review_sections.iterrows():
                for word in hidden_params[review_id][section_og_id].keys():
                    word_id = inverse_vocab_lookup[strip_punctuation(word)]
                    actual_param = dense_hidden_back_params[section_row["section_id"], word_id]
                    original_param = hidden_back_params[review_id][section_og_id][word]

                    print("checking word:" + word)
                    self.assertEqual(round(actual_param, 8), round(original_param, 8),
                                     msg="hidden background: " + ", word=" + word + ", a=" + str(
                                         actual_param) + ", e=" + str(original_param))

                section_og_id += 1

        # Check pi parameters
        for review_id in range(0, len(pi_params)):
            review_sections = pm_inst.parsed_text["section_list"]
            review_sections = review_sections[review_sections.doc_id == review_id]
            section_og_id = 0
            for section_index, section_row in review_sections.iterrows():
                for feature_id in range(0, len(pi_params[review_id][section_og_id])):
                    feature_name_row = pm_inst.formatted_feature_list[
                        pm_inst.formatted_feature_list.feature_id == feature_id]
                    feature_name_row = feature_name_row.reset_index(drop=True)
                    actual_param = em.pi_matrix[section_index, feature_id]
                    original_param = pi_params[review_id][section_og_id][feature_name_row["feature"][0]]

                    print("checking section:" + str(section_index))
                    self.assertEqual(round(actual_param, 8), round(original_param, 8),
                                     msg="pi params: " + ", feature=" + str(feature_name_row["feature"][0]) +
                                         ", section= " + str(section_index) + ", a=" + str(
                                         actual_param) + ", e=" + str(original_param))

                section_og_id += 1

        # Check pi deltas
        self.assertEqual(round(pi_delta, 8), round(em.pi_delta, 8),
                         msg="pi delta: " + ", a=" + str(
                             em.pi_delta) + ", e=" + str(pi_delta))