# -*- coding: utf-8 -*-

from unittest import TestCase
from feature_mining import ParseAndModel
import numpy as np
import pandas as pd
from collections import defaultdict, OrderedDict, Counter
from scipy.sparse import csr_matrix


class TestParseAndModel(TestCase):

    def test_format_feature_list_basic(self):
        pm = ParseAndModel()

        df = pd.DataFrame([["sound", 0, 0],
                           ["battery", 1, 1],
                           ["screen", 2, 2],
                           ["display", 3, 3]], columns=["feature", "feature_id", "feature_term_id"])

        feature_list = ["sound", "battery", "screen", "display"]


        pm.feature_list = feature_list
        pm.format_feature_list()

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
        pm.format_feature_list()

        print(df)
        print(pm.formatted_feature_list)
        self.assertEqual(True, pd.DataFrame.equals(df, pm.formatted_feature_list))

    def test_read_annotated_dat_one_line(self):
        pm = ParseAndModel()

        df_section_list = pd.DataFrame([[0, 0, "very pleased", True]],
                                       columns=["doc_id", "section_id", "section_text", "title"])
        df_feature_mapping = pd.DataFrame([])
        df_feature_list = defaultdict(int)

        pm.read_annotated_data(filename='data/parse_and_model/iPod.final', nlines=1)

        self.assertEqual(True, pd.DataFrame.equals(df_section_list, pm.parsed_text["section_list"]))
        self.assertEqual(True, pd.DataFrame.equals(df_feature_mapping, pm.parsed_text["feature_mapping"]))
        # self.assertEqual(0, ((df_feature_list > oneLine["feature_list"]) - (df_feature_list < oneLine["feature_list"])))

    def test_read_annotated_dat_one_feature_implicit(self):
        pm = ParseAndModel()

        df_section_list = pd.DataFrame([[0, 0, "it is handy to carry around because of the  and easy to store", False]],
                                       columns=["doc_id", "section_id", "section_text", "title"])
        df_feature_mapping = pd.DataFrame([[0, "size", False, 0]],
                                          columns=["doc_id", "feature", "is_explicit", "section_id"])
        df_feature_list = defaultdict(int)

        pm.read_annotated_data(filename='data/parse_and_model/iPod.final', nlines=1,
                                                      start_line=6)

        self.assertEqual(True, pd.DataFrame.equals(df_section_list, pm.parsed_text["section_list"]))
        self.assertEqual(True, pd.DataFrame.equals(df_feature_mapping, pm.parsed_text["feature_mapping"]))
        # self.assertEqual(0, ((df_feature_list > one_line["feature_list"]) - (df_feature_list < one_line["feature_list"])))

    def test_read_annotated_dat_one_feature_explicit(self):
        pm = ParseAndModel()

        df_section_list = pd.DataFrame([[0, 0, "the battery life is outstanding (again, compared to the mini)", False]],
                                       columns=["doc_id", "section_id", "section_text", "title"])
        df_feature_mapping = pd.DataFrame([[0, "battery", True, 0]],
                                          columns=["doc_id", "feature", "is_explicit", "section_id"])
        df_feature_list = defaultdict(int)
        df_feature_list["battery"] = 1

        pm.read_annotated_data(filename='data/parse_and_model/iPod.final', nlines=1,
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

        pm.read_annotated_data(filename='data/parse_and_model/iPod.final', nlines=1,
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
                                           , [1, 6, "i originally picked a silver one, because that is all the store had",
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

        pm.read_annotated_data(filename='data/parse_and_model/iPod.final', nlines=10,
                                                      start_line=660)

        self.assertEqual(True, pd.DataFrame.equals(df_section_list, pm.parsed_text["section_list"]))
        self.assertEqual(True, pd.DataFrame.equals(df_feature_mapping, pm.parsed_text["feature_mapping"]))
        self.assertEqual(True, dict(df_feature_list) == dict(pm.parsed_text["feature_list"]))

    def test_bem_one_section(self):
        pm = ParseAndModel()

        section_list = pd.DataFrame([[0, 0, "large clear screen", True]
                                        ], columns=["doc_id", "section_id", "section_text", "title"])

        pm.feature_list = ["screen"]
        pm.format_feature_list()

        pm.parsed_text =  dict(section_list= section_list)
        pm.build_explicit_models()

        expected_model_background=[1/3, 1/3, 1/3]
        expected_model_feature=[[1/3, 1/3, 1/3]]
        expected_section_word_counts={0:Counter({"large": 1, "clear": 1, "screen":1})}
        expected_section_word_counts_matrix=[[1,1,1]]
        expected_model_background_matrix=np.array([1/3, 1/3, 1/3])
        expected_model_feature_matrix = np.array([[1 / 3], [1 / 3], [1 / 3]])
        expected_vocab_lookup={0: 'large', 1: 'clear', 2: 'screen'}

        self.assertEqual(True, expected_model_background== pm.model_results["model_background"])
        self.assertEqual(True, expected_model_feature == pm.model_results["model_feature"])
        #self.assertEqual(True, expected_section_word_counts == em_input["section_word_counts"])
        self.assertEqual(True, np.array_equiv(expected_section_word_counts_matrix, csr_matrix.toarray(pm.model_results["section_word_counts_matrix"])))
        self.assertEqual(True, np.array_equiv(expected_model_background_matrix,csr_matrix.toarray(pm.model_results["model_background_matrix"])))
        self.assertEqual(True, np.array_equiv(expected_model_feature_matrix, csr_matrix.toarray(pm.model_results["model_feature_matrix"])))
        self.assertEqual(True, expected_vocab_lookup == pm.model_results["vocabulary_lookup"])

    def test_bem_two_section(self):
        pm = ParseAndModel()

        section_list = pd.DataFrame([[0, 0, "large clear screen", True]
                                        , [0, 1, "large broken bad", True]
                                        ], columns=["doc_id", "section_id", "section_text", "title"])

        pm.feature_list = ["screen"]
        pm.format_feature_list()

        pm.parsed_text = dict(section_list=section_list)
        pm.build_explicit_models(lemmatize_words=False)

        expected_model_background = [1 / 3, 1 / 6, 1 / 6, 1 / 6, 1 / 6]
        expected_model_feature = [[0.218, 0.282, 0.282, 0.109, 0.109]]
        expected_section_word_counts = {0: Counter({"large": 1, "clear": 1, "screen": 1})
                                        ,1: Counter({"large": 1, "broken":1, "bad":1})}
        expected_section_word_counts_matrix = [[1, 1, 1, 0, 0]
                                              ,[1, 0, 0, 1, 1]]
        expected_model_background_matrix = np.array([1 / 3, 1 / 6, 1 / 6, 1 / 6, 1 / 6])
        expected_model_feature_matrix = np.array([[0.218], [0.282], [0.282], [0.109], [0.109]])
        expected_vocab_lookup = {0: 'large', 1: 'clear', 2: 'screen', 3: 'broken', 4:'bad'}

        self.assertEqual(True, expected_model_background == pm.model_results["model_background"])
        self.assertEqual(True, expected_model_feature == [[round(val, 3) for val in feature_model] for feature_model in pm.model_results["model_feature"]])
        #self.assertEqual(True, expected_section_word_counts == em_input["section_word_counts"])
        self.assertEqual(True,
                         np.array_equiv(expected_section_word_counts_matrix, csr_matrix.toarray(pm.model_results["section_word_counts_matrix"])))
        self.assertEqual(True, np.array_equiv(expected_model_background_matrix, csr_matrix.toarray(pm.model_results["model_background_matrix"])))
        self.assertEqual(True, np.array_equiv(expected_model_feature_matrix, np.round(csr_matrix.toarray(pm.model_results["model_feature_matrix"]),3)))
        self.assertEqual(True, expected_vocab_lookup == pm.model_results["vocabulary_lookup"])


# added for exploratory testing
if __name__ == '__main__':
    #pm = ParseAndModel()

    # format_feature_list, read_annotated_data, build_explicit_models
    feature_list = ParseAndModel.format_feature_list(feature_list=["sound", "battery", ["screen", "display"]])
    pass
