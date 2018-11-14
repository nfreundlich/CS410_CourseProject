from unittest import TestCase
from feature_mining import parse_and_model
import pandas as pd
from collections import defaultdict, OrderedDict


class TestParseAndModel(TestCase):

    def test_format_feature_list_basic(self):
        df = pd.DataFrame([["sound", 0, 0],
                           ["battery", 1, 1],
                           ["screen", 2, 2],
                           ["display", 3, 3]], columns=["feature", "feature_id", "feature_term_id"])

        feature_list = parse_and_model.format_feature_list(feature_list=
                                                           ["sound",
                                                            "battery",
                                                            "screen",
                                                            "display"]
                                                           )
        print(df)
        print(feature_list)
        self.assertEqual(True, pd.DataFrame.equals(df, feature_list))

    def test_format_feature_list_synonym(self):
        df = pd.DataFrame([["sound", 0, 0],
                           ["battery", 1, 1],
                           ["screen", 2, 2],
                           ["display", 2, 3]], columns=["feature", "feature_id", "feature_term_id"])

        feature_list = parse_and_model.format_feature_list(feature_list=
                                                           ["sound",
                                                            "battery",
                                                            ["screen", "display"]]
                                                           )
        print(df)
        print(feature_list)
        self.assertEqual(True, pd.DataFrame.equals(df, feature_list))

    def test_read_annotated_dat_one_line(self):
        df_section_list = pd.DataFrame([[0, 0, "very pleased", True]], columns=["doc_id", "section_id", "section_text", "title"])
        df_feature_mapping = pd.DataFrame([])
        df_feature_list = defaultdict(int)

        oneLine = parse_and_model.read_annotated_data(filename='data/parse_and_model/iPod.final', nlines=1)

        self.assertEqual(True, pd.DataFrame.equals(df_section_list, oneLine["section_list"]))
        self.assertEqual(True, pd.DataFrame.equals(df_feature_mapping, oneLine["feature_mapping"]))
        #self.assertEqual(0, ((df_feature_list > oneLine["feature_list"]) - (df_feature_list < oneLine["feature_list"])))

    def test_read_annotated_dat_one_feature_implicit(self):
        df_section_list = pd.DataFrame([[0, 0, "it is handy to carry around because of the  and easy to store", False]], columns=["doc_id", "section_id", "section_text", "title"])
        df_feature_mapping = pd.DataFrame([[0, "size", False, 0]],
                                          columns=["doc_id", "feature", "is_explicit", "section_id"])
        df_feature_list = defaultdict(int)

        oneLine = parse_and_model.read_annotated_data(filename='data/parse_and_model/iPod.final', nlines=1, start_line=6)

        self.assertEqual(True, pd.DataFrame.equals(df_section_list, oneLine["section_list"]))
        self.assertEqual(True, pd.DataFrame.equals(df_feature_mapping, oneLine["feature_mapping"]))
        #self.assertEqual(0, ((df_feature_list > oneLine["feature_list"]) - (df_feature_list < oneLine["feature_list"])))

    def test_read_annotated_dat_one_feature_explicit(self):
        df_section_list = pd.DataFrame([[0, 0, "the battery life is outstanding (again, compared to the mini)", False]], columns=["doc_id", "section_id", "section_text", "title"])
        df_feature_mapping = pd.DataFrame([[0, "battery", True, 0]], columns=["doc_id", "feature", "is_explicit", "section_id"])
        df_feature_list = defaultdict(int)
        df_feature_list["battery"] = 1

        oneLine = parse_and_model.read_annotated_data(filename='data/parse_and_model/iPod.final', nlines=1, start_line=13)

        self.assertEqual(True, pd.DataFrame.equals(df_section_list, oneLine["section_list"]))
        self.assertEqual(True, pd.DataFrame.equals(df_feature_mapping, oneLine["feature_mapping"]))
        self.assertEqual(True, dict(df_feature_list) == dict(oneLine["feature_list"]))

    def test_read_annotated_dat_two_feature_explicit(self):
        df_section_list = pd.DataFrame([[0, 0, "my son loves the nano, it is small and has a good size screen", False]], columns=["doc_id", "section_id", "section_text", "title"])
        df_feature_mapping = pd.DataFrame([[0, "screen", True, 0]
                                           ,[0, "size", True, 0]], columns=["doc_id", "feature", "is_explicit", "section_id"])
        df_feature_list = defaultdict(int)
        df_feature_list["screen"]=1
        df_feature_list["size"] = 1

        oneLine = parse_and_model.read_annotated_data(filename='data/parse_and_model/iPod.final', nlines=1, start_line=622)

        self.assertEqual(True, pd.DataFrame.equals(df_section_list, oneLine["section_list"]))
        self.assertEqual(True, pd.DataFrame.equals(df_feature_mapping, oneLine["feature_mapping"]))
        self.assertEqual(True, dict(df_feature_list) == dict(oneLine["feature_list"]))

    def test_read_annotated_dat_complicated_reviews(self):
        df_section_list = pd.DataFrame([[0, 0, "it could be better", True]
                                            ,   [0, 1, "this item is very nice and plays songs in a very good stereo sound but my problem with this item is the  it does not last not even near the 14 hours they claimed to last", False]
                                            ,   [0, 2, "i hope the new nano is at leat close to the hours claimed", False]
                                            ,   [1, 3, "pink apple 4gb nano ipod review", True]
                                            ,   [1, 4, "it was a gift for christmas to my daughter", False]
                                            ,   [1, 5, "she absolutely loves it! the sound quality is excellent!! the different colors make a nice option as well", False]
                                            ,   [1, 6, "i originally picked a silver one, because that is all the store had", False]
                                            ,   [1, 7, "i then checked amazon", False]
                                            ,   [1, 8, "com that not only had it in pink(the color my daughter wanted), [", False]
                                            ,   [1, 9, "]i would recommend this product to everyone", False]

                                        ], columns=["doc_id", "section_id", "section_text", "title"])
        df_feature_mapping = pd.DataFrame([[0, "battery", False, 1]
                                           , [0, "sound", True, 1]
                                           , [1, "sound", True, 5]], columns=["doc_id", "feature", "is_explicit", "section_id"])
        df_feature_list = defaultdict(int)
        df_feature_list["battery"] = 1
        df_feature_list["sound"] = 2

        oneLine = parse_and_model.read_annotated_data(filename='data/parse_and_model/iPod.final', nlines=10, start_line=660)

        self.assertEqual(True, pd.DataFrame.equals(df_section_list, oneLine["section_list"]))
        self.assertEqual(True, pd.DataFrame.equals(df_feature_mapping, oneLine["feature_mapping"]))
        self.assertEqual(True, dict(df_feature_list) == dict(oneLine["feature_list"]))

    def test_build_explicit_models(self):
        self.fail()


# added for exploratory testing
if __name__ == '__main__':
    # format_feature_list, read_annotated_data, build_explicit_models
    feature_list = parse_and_model.format_feature_list(feature_list=["sound", "battery", ["screen", "display"]])
    pass
