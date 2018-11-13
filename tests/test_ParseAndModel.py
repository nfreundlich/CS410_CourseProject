from unittest import TestCase
from feature_mining import parse_and_model
import pandas as pd


class TestParseAndModel(TestCase):
    def test_format_feature_list(self):
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

    def test_read_annotated_data(self):
        self.fail()

    def test_build_explicit_models(self):
        self.fail()


# added for exploratory testing
if __name__ == '__main__':
    # format_feature_list, read_annotated_data, build_explicit_models
    feature_list = parse_and_model.format_feature_list(feature_list=["sound", "battery", ["screen", "display"]])
    pass
