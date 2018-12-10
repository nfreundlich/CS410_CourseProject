# -*- coding: utf-8 -*-

from unittest import TestCase
from feature_mining.gflm_tagger import GFLM
from feature_mining.em_vector_by_feature import EmVectorByFeature
from feature_mining import ParseAndModel
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict, OrderedDict, Counter
from scipy.sparse import csr_matrix


def strip_punctuation(s):
    new_s = ''.join(c for c in s if c not in string.punctuation)
    return new_s.strip()


class TestGFLM(TestCase):

    def test_constructor(self):
        pass

    def test_small_results(self):
        pi_matrix = np.array([[.3, .7], [.6, .4]])
        hidden_background = csr_matrix(np.array([[.8, .6, .1], [.1, .2, .3]]))
        hidden_params = dict()
        hidden_params[0] = csr_matrix(np.array([[.5, .4, .1], [.9, .8, .3]]))
        hidden_params[1] = csr_matrix(np.array([[.5, .6, .9], [.1, .2, .7]]))

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

    def test_against_original_1_50_iteration(self):

        # load pi initalization
        infile = open("original_code_data/test_original_1_pi_init.data", 'rb')
        pi_init = pickle.load(infile)
        infile.close()

        # load gflm sentence calculation results
        infile = open("original_code_data/test_original_1_gflm_sentence_probs.data", 'rb')
        gflm_sentence_probs = pickle.load(infile)
        infile.close()

        # load gflm sentence tagging results
        infile = open("original_code_data/test_original_1_gflm_sentence_results.data", 'rb')
        gflm_sentence_results = pickle.load(infile)
        infile.close()

        # load gflm word calculation result
        infile = open("original_code_data/test_original_1_gflm_word_probs.data", 'rb')
        gflm_word_probs = pickle.load(infile)
        infile.close()

        # load gflm word tagging results
        infile = open("original_code_data/test_original_1_gflm_word_results.data", 'rb')
        gflm_word_results = pickle.load(infile)
        infile.close()

        pm_inst = ParseAndModel(feature_list=["sound", "battery"], filename='data/parse_and_model/iPod.final',
                                remove_stopwords=False, lemmatize_words=False, log_base=None, start_line=4, nlines=11,
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

        em = EmVectorByFeature(explicit_model=pm_inst, max_iter=50, lambda_background=0.7, pi_init=pi_init_em)
        em.initialize_parameters()
        em.em_loop()

        # Calculate GFLM
        gflm = GFLM(em_results=em, section_threshold=0.35, word_threshold=0.35)
        gflm.calc_gflm_section()
        gflm.calc_gflm_word()

        # Check gflm word results
        for review_id in range(0, len(gflm_word_probs)):
            review_sections = pm_inst.parsed_text["section_list"]
            review_sections = review_sections[review_sections.doc_id == review_id]
            section_og_id = 0
            for section_index, section_row in review_sections.iterrows():
                for feature_id in range(0, max(gflm.gflm_word_all.implicit_feature_id)+1):
                    feature_name_row = pm_inst.formatted_feature_list[
                        pm_inst.formatted_feature_list.feature_id == feature_id]
                    feature_name_row = feature_name_row.reset_index(drop=True)
                    actual_param_data = gflm.gflm_word_all[(gflm.gflm_word_all.section_id == section_index) & (gflm.gflm_word_all.implicit_feature_id == feature_id)]
                    actual_param_data = actual_param_data.reset_index(drop=True)
                    actual_param = actual_param_data.loc[0].gflm_word

                    # loop through words and grab relevant feature value
                    word_probs = list()
                    for word, probability in gflm_word_probs[review_id][section_og_id].items():
                        word_probs.append(probability[feature_name_row["feature"][0]])
                    original_param = max(word_probs)

                    print("checking GFLM word probs - section:" + str(section_index), ", feature=" + str(feature_id))
                    self.assertEqual(round(actual_param, 8), round(original_param, 8),
                                     msg="feature=" + str(feature_name_row["feature"][0]) +
                                         ", section= " + str(section_index) + ", a=" + str(
                                         actual_param) + ", e=" + str(original_param))

                section_og_id += 1

        # Check gflm section results
        for review_id in range(0, len(gflm_word_probs)):
            review_sections = pm_inst.parsed_text["section_list"]
            review_sections = review_sections[review_sections.doc_id == review_id]
            section_og_id = 0
            for section_index, section_row in review_sections.iterrows():
                for feature_id in range(0, max(gflm.gflm_word_all.implicit_feature_id)+1):
                    feature_name_row = pm_inst.formatted_feature_list[
                        pm_inst.formatted_feature_list.feature_id == feature_id]
                    feature_name_row = feature_name_row.reset_index(drop=True)
                    actual_param_data = gflm.gflm_section_all[(gflm.gflm_section_all.section_id == section_index) & (gflm.gflm_section_all.implicit_feature_id == feature_id)]
                    actual_param_data = actual_param_data.reset_index(drop=True)
                    actual_param = actual_param_data.loc[0].gflm_section

                    original_param = gflm_sentence_probs[review_id][section_og_id][feature_name_row["feature"][0]]

                    print("checking GFLM section probs - section:" + str(section_index), ", feature=" + str(feature_id))
                    self.assertEqual(round(actual_param, 8), round(original_param, 8),
                                     msg="feature=" + str(feature_name_row["feature"][0]) +
                                         ", section= " + str(section_index) + ", a=" + str(
                                         actual_param) + ", e=" + str(original_param))

                section_og_id += 1


# added for exploratory testing
if __name__ == '__main__':
    pass
