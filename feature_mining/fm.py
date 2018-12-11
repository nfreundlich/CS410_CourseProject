import logging
from enum import Enum
import pkg_resources
from feature_mining import ParseAndModel
from feature_mining import EmVectorByFeature
from feature_mining import GFLM


class FeatureMining:
    """
    Workflow for implicit feature mining.
    Usage:
        from feature_mining import FeatureMining
        fm.usage()
    """
    def __init__(self):
        """
        Constructor. Initialize members to None.
        """
        logging.info(type(self).__name__, "- init...")
        self.pm = None
        self.em = None
        self.gflm = None
        self.feature_list = None

    def load_data(self,
                  feature_list: list = None,
                  filename: str = None,
                  input_type: Enum = ParseAndModel.InputType.annotated,
                  nlines: int = None, ):
        """
        Loads user data.
        :param feature_list: a list of strings and lists of strings. Individual strings will be given separate ids, lists
        of strings will be treated as synonyms and given the same feature id.
        ex. ["sound", "battery", ["screen", "display"]]
        :param filename: Filename for the data set
        :param input_type: An enum of type InputType, specifying the type of input data so the correct read function can be chosen
            options are "annotated" - which expects data in Santu's original format and "onedocperline" - which expects
            all data to be in a single file with one document per line
        :param nlines: Maximum number of lines from the file to read or None to read all lines
        :return: None
        """
        logging.info(type(self).__name__, "- load_data...")
        if not feature_list:
            self.usage()
            exit(1)
        if not filename:
            self.usage()
            exit(1)
        self.pm = ParseAndModel(feature_list=feature_list,  # list of features
                                filename=filename,  # file with input data
                                nlines=nlines)  # number of lines to read

    def load_ipod(self, full_set: bool = False):
        """
        Loads sample data.
        :param full_set: (optional) Bool setting whether to load the entire dataset. If True, computing will take several seconds.
        :return: None
        """
        logging.info(type(self).__name__, "- load_ipod...")

        # data_path = pkg_resources.resource_filename('feature_mining', 'data/')
        filename = pkg_resources.resource_filename('feature_mining', 'data/iPod.final')

        logging.warning(">loading file:" + filename)

        feature_list = ["sound", "battery", ["screen", "display"]]
        nlines = None

        if not full_set:
            nlines = 300

        self.load_data(feature_list=feature_list,  # list of features
                       filename=filename,  # file with input data
                       input_type=ParseAndModel.InputType.annotated, # data format in input file
                       nlines=nlines)  # number of lines to read

        print("loaded features: ", feature_list)
        print("loaded dataset: ipod")

        logging.info(type(self).__name__, "- done: load_ipod...")

    def fit(self, max_iter=50):
        """
        Executes Expectation-Maximization on previously loaded data.
        :param max_iter: maximum number of iterations in Expectation Maximization
        :return: None
        """
        logging.info(type(self).__name__, "- fit...")
        self.em = EmVectorByFeature(explicit_model=self.pm,
                                    max_iter=max_iter)
        self.em.em()
        logging.info(type(self).__name__, "- done: fit...")

    def predict(self, section_threshold=0.35, word_threshold=0.35):
        """
        Executes GFLM section and word.
        :param section_threshold: threshold to be used for gflm section
        :param word_threshold: threshold to be used for gflm word
        :return: None
        """
        logging.info(type(self).__name__, "- predict...")
        self.gflm = GFLM(em_results=self.em, section_threshold=0.35, word_threshold=0.35)
        self.gflm.calc_gflm_section()
        self.gflm.calc_gflm_word()

    def usage(self):
        """
        Prints usage message.
        :return: None
        """
        USAGE = """
        Implicit feature mining.
        
        Usage:
            from feature_mining import FeatureMining
            fm = FeatureMining()
            fm.load_ipod(full_set=False)
            fm.fit()
            fm.predict()
        
        Results:
            - prediction using 'section': fm.gflm.gflm_section
            - prediction using 'word': fm.gflm.gflm_word
        """
        print(USAGE)


if __name__ == '__main__':
    fm = FeatureMining()
    fm.load_ipod(full_set=False)
    fm.fit()
    fm.predict()
    print(fm.gflm.gflm_section.head(10))
    print(fm.gflm.gflm_word.head(10))
