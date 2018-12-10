# -*- coding: utf-8 -*-

from unittest import TestCase
from feature_mining import ParseAndModel
import numpy as np
import pandas as pd
from collections import defaultdict, OrderedDict, Counter
from scipy.sparse import csr_matrix
import pickle
import os
import string
import math


def strip_punctuation(s):
    new_s = ''.join(c for c in s if c not in string.punctuation)
    return new_s.strip()


class TestParseAndModel(TestCase):

    def test_format_feature_list_basic(self):
        pm = ParseAndModel()

        df = pd.DataFrame([["sound", 0, 0],
                           ["battery", 1, 1],
                           ["screen", 2, 2],
                           ["display", 3, 3]], columns=["feature", "feature_id", "feature_term_id"])

        pm.formatted_feature_list = feature_list = ["sound", "battery", "screen", "display"]

        pm.feature_list = feature_list
        pm.formatted_feature_list = pm.format_feature_list()

        print(df)
        print(pm.formatted_feature_list)
        self.assertEqual(True, pd.DataFrame.equals(df, pm.formatted_feature_list))

    def test_format_feature_list_synonym(self):
        pm = ParseAndModel()

        df = pd.DataFrame([["sound", 0, 0],
                           ["battery", 1, 1],
                           ["screen", 2, 2],
                           ["display", 2, 3]], columns=["feature", "feature_id", "feature_term_id"])

        feature_list = ["sound", "battery", ["screen", "display"]]

        pm.feature_list = feature_list
        pm.formatted_feature_list = pm.format_feature_list()

        print(df)
        print(pm.formatted_feature_list)
        self.assertEqual(True, pd.DataFrame.equals(df, pm.formatted_feature_list))

    def test_read_plain_dat_one_line(self):
        pm = ParseAndModel(filename='../tests/data/parse_and_model/oneLinePerDoc.txt', input_type="oneDocPerLine", nlines=1)

        df_section_list = pd.DataFrame([[0, 0, "I am very pleased with the 4 GB iPod Nano that I purchased."],
                                        [0, 1, "It was very easy to download music onto it and it's very easy to move around in it."],
                                        [0, 2, "Recommend this item to anybody."]],
                                       columns=["doc_id", "section_id", "section_text"])

        self.assertEqual(True, pd.DataFrame.equals(df_section_list, pm.parsed_text["section_list"]))

    def test_read_plain_dat_two_line(self):
        pm = ParseAndModel(filename='../tests/data/parse_and_model/oneLinePerDoc.txt', input_type="oneDocPerLine", nlines=2)

        df_section_list = pd.DataFrame([[0, 0, "I am very pleased with the 4 GB iPod Nano that I purchased."],
                                        [0, 1, "It was very easy to download music onto it and it's very easy to move around in it."],
                                        [0, 2, "Recommend this item to anybody."],
                                        [1, 3, "I like the compact ipod and the features it offered."],
                                        [1, 4, "It is handy to carry around because of the  and easy to store."],
                                        [1, 5, "The light weight also makes it easy to move with."],
                                        [1, 6, "It works well and I have had no problems with it."]],
                                       columns=["doc_id", "section_id", "section_text"])

        self.assertEqual(True, pd.DataFrame.equals(df_section_list, pm.parsed_text["section_list"]))

    def test_read_plain_dat_three_line(self):
        pm = ParseAndModel(filename='../tests/data/parse_and_model/oneLinePerDoc.txt', input_type="oneDocPerLine")

        df_section_list = pd.DataFrame([[0, 0, "I am very pleased with the 4 GB iPod Nano that I purchased."],
                                        [0, 1, "It was very easy to download music onto it and it's very easy to move around in it."],
                                        [0, 2, "Recommend this item to anybody."],
                                        [1, 3, "I like the compact ipod and the features it offered."],
                                        [1, 4, "It is handy to carry around because of the  and easy to store."],
                                        [1, 5, "The light weight also makes it easy to move with."],
                                        [1, 6, "It works well and I have had no problems with it."],
                                        [2, 7, "This is my second iPod."],
                                        [2, 8, 'My first was a "mini" which the nano makes look like a "jumbo".'],
                                        [2, 9, "It's very lightweight, sound quality is typical of these devices."],
                                        [2, 10, "The battery life is outstanding (again, compared to the mini)."],
                                        [2, 11, "I've only had it for a month, but the battery so far is lasting over 8 hours."],
                                        [2, 12,
                                         "I haven't completely run it until it is dead yet, so I don't know how long it will really last."],
                                        [2, 13,"Awesome!"],
                                        ],
                                       columns=["doc_id", "section_id", "section_text"])

        self.assertEqual(True, pd.DataFrame.equals(df_section_list, pm.parsed_text["section_list"]))

    def test_read_annotated_dat_one_line(self):
        pm = ParseAndModel()

        df_section_list = pd.DataFrame([[0, 0, "very pleased", True]],
                                       columns=["doc_id", "section_id", "section_text", "title"])
        df_feature_mapping = pd.DataFrame([])
        df_feature_list = defaultdict(int)

        pm.parsed_text = pm.read_annotated_data(filename='data/parse_and_model/iPod.final', nlines=1)

        self.assertEqual(True, pd.DataFrame.equals(df_section_list, pm.parsed_text["section_list"]))
        self.assertEqual(True, pd.DataFrame.equals(df_feature_mapping, pm.parsed_text["feature_mapping"]))

    def test_read_annotated_dat_one_feature_implicit(self):
        pm = ParseAndModel()

        df_section_list = pd.DataFrame([[0, 0, "it is handy to carry around because of the  and easy to store", False]],
                                       columns=["doc_id", "section_id", "section_text", "title"])
        df_feature_mapping = pd.DataFrame([[0, "size", False, 0]],
                                          columns=["doc_id", "feature", "is_explicit", "section_id"])
        df_feature_list = defaultdict(int)

        pm.parsed_text = pm.read_annotated_data(filename='data/parse_and_model/iPod.final', nlines=1,
                                                start_line=6)

        self.assertEqual(True, pd.DataFrame.equals(df_section_list, pm.parsed_text["section_list"]))
        self.assertEqual(True, pd.DataFrame.equals(df_feature_mapping, pm.parsed_text["feature_mapping"]))

    def test_read_annotated_dat_one_feature_explicit(self):
        pm = ParseAndModel()

        df_section_list = pd.DataFrame([[0, 0, "the battery life is outstanding (again, compared to the mini)", False]],
                                       columns=["doc_id", "section_id", "section_text", "title"])
        df_feature_mapping = pd.DataFrame([[0, "battery", True, 0]],
                                          columns=["doc_id", "feature", "is_explicit", "section_id"])
        df_feature_list = defaultdict(int)
        df_feature_list["battery"] = 1

        pm.parsed_text = pm.read_annotated_data(filename='data/parse_and_model/iPod.final', nlines=1,
                                                start_line=13)

        self.assertEqual(True, pd.DataFrame.equals(df_section_list, pm.parsed_text["section_list"]))
        self.assertEqual(True, pd.DataFrame.equals(df_feature_mapping, pm.parsed_text["feature_mapping"]))
        self.assertEqual(True, dict(df_feature_list) == dict(pm.parsed_text["feature_list"]))

    def test_read_annotated_dat_two_feature_explicit(self):
        pm = ParseAndModel()

        df_section_list = pd.DataFrame([[0, 0, "my son loves the nano, it is small and has a good size screen", False]],
                                       columns=["doc_id", "section_id", "section_text", "title"])
        df_feature_mapping = pd.DataFrame([[0, "screen", True, 0]
                                              , [0, "size", True, 0]],
                                          columns=["doc_id", "feature", "is_explicit", "section_id"])
        df_feature_list = defaultdict(int)
        df_feature_list["screen"] = 1
        df_feature_list["size"] = 1

        pm.parsed_text = pm.read_annotated_data(filename='data/parse_and_model/iPod.final', nlines=1,
                                                start_line=622)

        self.assertEqual(True, pd.DataFrame.equals(df_section_list, pm.parsed_text["section_list"]))
        self.assertEqual(True, pd.DataFrame.equals(df_feature_mapping, pm.parsed_text["feature_mapping"]))
        self.assertEqual(True, dict(df_feature_list) == dict(pm.parsed_text["feature_list"]))

    def test_read_annotated_dat_complicated_reviews(self):
        pm = ParseAndModel()

        df_section_list = pd.DataFrame([[0, 0, "it could be better", True]
                                           , [0, 1,
                                              "this item is very nice and plays songs in a very good stereo sound but my problem with this item is the  it does not last not even near the 14 hours they claimed to last",
                                              False]
                                           , [0, 2, "i hope the new nano is at leat close to the hours claimed", False]
                                           , [1, 3, "pink apple 4gb nano ipod review", True]
                                           , [1, 4, "it was a gift for christmas to my daughter", False]
                                           , [1, 5,
                                              "she absolutely loves it! the sound quality is excellent!! the different colors make a nice option as well",
                                              False]
                                           ,
                                        [1, 6, "i originally picked a silver one, because that is all the store had",
                                         False]
                                           , [1, 7, "i then checked amazon", False]
                                           , [1, 8, "com that not only had it in pink(the color my daughter wanted), [",
                                              False]
                                           , [1, 9, "]i would recommend this product to everyone", False]

                                        ], columns=["doc_id", "section_id", "section_text", "title"])
        df_feature_mapping = pd.DataFrame([[0, "battery", False, 1]
                                              , [0, "sound", True, 1]
                                              , [1, "sound", True, 5]],
                                          columns=["doc_id", "feature", "is_explicit", "section_id"])
        df_feature_list = defaultdict(int)
        df_feature_list["battery"] = 1
        df_feature_list["sound"] = 2

        pm.parsed_text = pm.read_annotated_data(filename='data/parse_and_model/iPod.final', nlines=10,
                                                start_line=660)

        self.assertEqual(True, pd.DataFrame.equals(df_section_list, pm.parsed_text["section_list"]))
        self.assertEqual(True, pd.DataFrame.equals(df_feature_mapping, pm.parsed_text["feature_mapping"]))
        self.assertEqual(True, dict(df_feature_list) == dict(pm.parsed_text["feature_list"]))

    def test_bem_one_section(self):
        pm = ParseAndModel()

        section_list = pd.DataFrame([[0, 0, "large clear screen", True]
                                     ], columns=["doc_id", "section_id", "section_text", "title"])

        pm.feature_list = ["screen"]
        pm.formatted_feature_list = pm.format_feature_list()

        pm.parsed_text = dict(section_list=section_list)
        pm.model_results = pm.build_explicit_models(log_base=2)

        expected_model_background = [1 / 3, 1 / 3, 1 / 3]
        expected_model_feature = [[1 / 3, 1 / 3, 1 / 3]]
        expected_section_word_counts = {0: Counter({"large": 1, "clear": 1, "screen": 1})}
        expected_section_word_counts_matrix = [[1, 1, 1]]
        expected_model_background_matrix = np.array([1 / 3, 1 / 3, 1 / 3])
        expected_model_feature_matrix = np.array([[1 / 3], [1 / 3], [1 / 3]])
        expected_vocab_lookup = {0: 'large', 1: 'clear', 2: 'screen'}

        self.assertEqual(True, expected_model_background == pm.model_results["model_background"])
        self.assertEqual(True, expected_model_feature == pm.model_results["model_feature"])
        # self.assertEqual(True, expected_section_word_counts == em_input["section_word_counts"])
        self.assertEqual(True, np.array_equiv(expected_section_word_counts_matrix,
                                              csr_matrix.toarray(pm.model_results["section_word_counts_matrix"])))
        self.assertEqual(True, np.array_equiv(expected_model_background_matrix,
                                              csr_matrix.toarray(pm.model_results["model_background_matrix"])))
        self.assertEqual(True, np.array_equiv(expected_model_feature_matrix, pm.model_results["model_feature_matrix"]))
        self.assertEqual(True, expected_vocab_lookup == pm.model_results["vocabulary_lookup"])

    def test_bem_two_section(self):
        pm = ParseAndModel()

        section_list = pd.DataFrame([[0, 0, "large clear screen", True]
                                        , [0, 1, "large broken bad", True]
                                     ], columns=["doc_id", "section_id", "section_text", "title"])

        pm.feature_list = ["screen"]
        pm.formatted_feature_list = pm.format_feature_list()

        pm.parsed_text = dict(section_list=section_list)
        pm.model_results = pm.build_explicit_models(lemmatize_words=False, log_base=2)

        expected_model_background = [1 / 3, 1 / 6, 1 / 6, 1 / 6, 1 / 6]
        expected_model_feature = [[0.218, 0.282, 0.282, 0.109, 0.109]]
        expected_section_word_counts = {0: Counter({"large": 1, "clear": 1, "screen": 1})
            , 1: Counter({"large": 1, "broken": 1, "bad": 1})}
        expected_section_word_counts_matrix = [[1, 1, 1, 0, 0]
            , [1, 0, 0, 1, 1]]
        expected_model_background_matrix = np.array([1 / 3, 1 / 6, 1 / 6, 1 / 6, 1 / 6])
        expected_model_feature_matrix = np.array([[0.218], [0.282], [0.282], [0.109], [0.109]])
        expected_vocab_lookup = {0: 'large', 1: 'clear', 2: 'screen', 3: 'broken', 4: 'bad'}

        self.assertEqual(True, expected_model_background == pm.model_results["model_background"])
        self.assertEqual(True, expected_model_feature == [[round(val, 3) for val in feature_model] for feature_model in
                                                          pm.model_results["model_feature"]])
        # self.assertEqual(True, expected_section_word_counts == em_input["section_word_counts"])
        self.assertEqual(True,
                         np.array_equiv(expected_section_word_counts_matrix,
                                        csr_matrix.toarray(pm.model_results["section_word_counts_matrix"])))
        self.assertEqual(True, np.array_equiv(expected_model_background_matrix,
                                              csr_matrix.toarray(pm.model_results["model_background_matrix"])))
        self.assertEqual(True, np.array_equiv(expected_model_feature_matrix,
                                              np.round(pm.model_results["model_feature_matrix"], 3)))
        self.assertEqual(True, expected_vocab_lookup == pm.model_results["vocabulary_lookup"])

    def test_constructor_one_section(self):
        pm = ParseAndModel(feature_list=["screen"], filename='data/parse_and_model/twoLineTest.txt',
                           lemmatize_words=False, nlines=1)

        section_list = pd.DataFrame([[0, 0, "large clear screen", True]
                                     ], columns=["doc_id", "section_id", "section_text", "title"])

        expected_model_background = [1 / 3, 1 / 3, 1 / 3]
        expected_model_feature = [[1 / 3, 1 / 3, 1 / 3]]
        expected_section_word_counts = {0: Counter({"large": 1, "clear": 1, "screen": 1})}
        expected_section_word_counts_matrix = [[1, 1, 1]]
        expected_model_background_matrix = np.array([1 / 3, 1 / 3, 1 / 3])
        expected_model_feature_matrix = np.array([[1 / 3], [1 / 3], [1 / 3]])
        expected_vocab_lookup = {0: 'large', 1: 'clear', 2: 'screen'}

        self.assertEqual(True, expected_model_background == pm.model_results["model_background"])
        self.assertEqual(True, expected_model_feature == pm.model_results["model_feature"])
        # self.assertEqual(True, expected_section_word_counts == em_input["section_word_counts"])
        self.assertEqual(True, np.array_equiv(expected_section_word_counts_matrix,
                                              csr_matrix.toarray(pm.model_results["section_word_counts_matrix"])))
        self.assertEqual(True, np.array_equiv(expected_model_background_matrix,
                                              csr_matrix.toarray(pm.model_results["model_background_matrix"])))
        self.assertEqual(True, np.array_equiv(expected_model_feature_matrix, pm.model_results["model_feature_matrix"]))
        self.assertEqual(True, expected_vocab_lookup == pm.model_results["vocabulary_lookup"])

    def test_constructor_two_section(self):
        pm = ParseAndModel(feature_list=["screen"], filename='data/parse_and_model/twoLineTest.txt',
                           lemmatize_words=False, log_base=2)

        section_list = pd.DataFrame([[0, 0, "large clear screen", True]
                                        , [0, 1, "large broken bad", True]
                                     ], columns=["doc_id", "section_id", "section_text", "title"])

        # pm.feature_list = ["screen"]
        # pm.format_feature_list()

        # pm.parsed_text = dict(section_list=section_list)
        # pm.build_explicit_models(lemmatize_words=False)

        expected_model_background = [1 / 3, 1 / 6, 1 / 6, 1 / 6, 1 / 6]
        expected_model_feature = [[0.218, 0.282, 0.282, 0.109, 0.109]]
        expected_section_word_counts = {0: Counter({"large": 1, "clear": 1, "screen": 1})
            , 1: Counter({"large": 1, "broken": 1, "bad": 1})}
        expected_section_word_counts_matrix = [[1, 1, 1, 0, 0]
            , [1, 0, 0, 1, 1]]
        expected_model_background_matrix = np.array([1 / 3, 1 / 6, 1 / 6, 1 / 6, 1 / 6])
        expected_model_feature_matrix = np.array([[0.218], [0.282], [0.282], [0.109], [0.109]])
        expected_vocab_lookup = {0: 'large', 1: 'clear', 2: 'screen', 3: 'broken', 4: 'bad'}

        self.assertEqual(True, expected_model_background == pm.model_results["model_background"])
        self.assertEqual(True,
                         expected_model_feature == [[round(val, 3) for val in feature_model] for feature_model in
                                                    pm.model_results["model_feature"]])
        # self.assertEqual(True, expected_section_word_counts == em_input["section_word_counts"])
        self.assertEqual(True,
                         np.array_equiv(expected_section_word_counts_matrix,
                                        csr_matrix.toarray(pm.model_results["section_word_counts_matrix"])))
        self.assertEqual(True, np.array_equiv(expected_model_background_matrix,
                                              csr_matrix.toarray(pm.model_results["model_background_matrix"])))
        self.assertEqual(True, np.array_equiv(expected_model_feature_matrix,
                                              np.round(pm.model_results["model_feature_matrix"],
                                                       3)))
        self.assertEqual(True, expected_vocab_lookup == pm.model_results["vocabulary_lookup"])

    def test_against_original_1(self):
        os.getcwd()

        # load topic model
        infile = open("original_code_data/test_original_1_topic_model.data", 'rb')
        topic_model = pickle.load(infile)
        infile.close()

        # load section word counts
        infile = open("original_code_data/test_original_1_section_word_counts.data", 'rb')
        section_word_counts = pickle.load(infile)
        infile.close()

        # load background model
        infile = open("original_code_data/test_original_1_background_model.data", 'rb')
        background_model = pickle.load(infile)
        infile.close()

        pm = ParseAndModel(feature_list=["sound", "battery"], filename='data/parse_and_model/iPod.final',
                           remove_stopwords=False, lemmatize_words=False, log_base=None, start_line=4, nlines=11,
                           include_title_lines=False)

        # check section word counts
        pm_section_word_counts = pm.model_results["section_word_counts_matrix"].toarray()
        inverse_vocab_lookup = {strip_punctuation(v): k for k, v in pm.model_results["vocabulary_lookup"].items()}
        for review_id in range(0, len(section_word_counts)):
            print("SWC - Checking review: " + str(review_id))
            review_sections = pm.parsed_text["section_list"]
            review_sections = review_sections[review_sections.doc_id == review_id]
            section_og_id = 0
            for section_index, section_row in review_sections.iterrows():
                print("SWC - Checking section:" + str(section_row["section_id"]))
                for word in section_word_counts[review_id][section_og_id].keys():
                    word = strip_punctuation(word)
                    if word == '':
                        continue
                    vocab_word_id = inverse_vocab_lookup[word]
                    actual_count = pm_section_word_counts[section_row["section_id"], vocab_word_id]
                    original_count = section_word_counts[review_id][section_og_id][word]

                    self.assertEqual(actual_count, original_count, msg="SWC - section_id: " + str(
                        section_row["section_id"]) + ", " + word + " a=" + str(actual_count) + ", e=" + str(
                        original_count))

                section_og_id += 1

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

                self.assertEqual(round(actual_prob, 8), round(original_prob, 8),
                                 msg="topic - feature_id: " + str(
                                     feature_row["feature_id"]) + ", word=" + word + ", a=" + str(
                                     actual_prob) + ", e=" + str(original_prob))


# added for exploratory testing
if __name__ == '__main__':
    # pm = ParseAndModel()

    # format_feature_list, read_annotated_data, build_explicit_models
    # feature_list = ParseAndModel.format_feature_list(feature_list=["sound", "battery", ["screen", "display"]])
    pass
