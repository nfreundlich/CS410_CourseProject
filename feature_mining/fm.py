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
        self.gflm_section_result = None
        self.gflm_word_result = None

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
                                nlines=nlines,  # number of lines to read
                                input_type=input_type)  # input type as enum

    def load_ipod(self, full_set: bool = False):
        """
        Loads sample data.
        :param full_set: (optional) Bool setting whether to load the entire dataset. If True, computing will take several seconds.
        :return: None
        """
        logging.info(type(self).__name__, "- load_ipod...")

        # data_path = pkg_resources.resource_filename('feature_mining', 'data/')
        filename = pkg_resources.resource_filename('feature_mining', 'data/iPod.final')

        logging.info(">loading file:" + filename)

        feature_list = ["sound",
                        "battery",
                        ["screen", "display"],
                        "storage",
                        "size",
                        "headphones",
                        "software",
                        "price",
                        "button",]
        nlines = None

        if not full_set:
            nlines = 300

        self.load_data(feature_list=feature_list,  # list of features
                       filename=filename,  # file with input data
                       input_type=ParseAndModel.InputType.annotated,  # data format in input file
                       nlines=nlines)  # number of lines to read

        print("loaded features: ", feature_list)
        print("loaded dataset: ipod")

        logging.info(type(self).__name__, "- done: load_ipod...")

    def fit(self, max_iter=50, delta_threshold=1e-6):
        """
        Executes Expectation-Maximization on previously loaded data.
        :param delta_threshold: delta threshold for stopping the iterations
        :param max_iter: maximum number of iterations in Expectation Maximization
        :return: None
        """
        logging.info(type(self).__name__, "- fit...")
        self.em = EmVectorByFeature(explicit_model=self.pm,
                                    max_iter=max_iter,
                                    delta_threshold=delta_threshold)
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
        self.gflm = GFLM(em_results=self.em, section_threshold=0.35, word_threshold=0.35,
                         explicit_feature_mapping=self.pm.model_results["feature_section_mapping"])
        self.gflm.calc_gflm_section()
        self.gflm.calc_gflm_word()

    def section_features(self):
        """
        Associates an implicit feature id with a specific section.
        :return: None
        """
        section_list = self.pm.parsed_text['section_list'][['section_id', 'section_text']]
        feature_list = self.pm.formatted_feature_list.drop_duplicates(subset=['feature_id'])
        gflm_word = self.gflm.gflm_word
        gflm_section = self.gflm.gflm_section

        # feature text equivalence
        feat_dict = {}
        for index, raw in feature_list.iterrows():
            feat_dict[raw['feature_id']] = raw['feature']


        # gflm_word
        gflm_word_join = gflm_word.join(section_list,
                                           rsuffix='_section_list',
                                           lsuffix='_gflm_word')
        gflm_word_join['feature'] = gflm_word_join['implicit_feature_id'].apply(lambda x: feat_dict[x])
        gflm_word_join = gflm_word_join[['feature', 'section_text', 'section_id_gflm_word', 'gflm_word']]

        self.gflm_word_result = gflm_word_join

        # gflm_section
        gflm_section_join = gflm_section.join(section_list, on='section_id',
                                              rsuffix='_section_list',
                                              lsuffix='_gflm_section')
        gflm_section_join['feature'] = gflm_section_join['implicit_feature_id'].apply(lambda x: feat_dict[x])
        gflm_section_join = gflm_section_join[['feature', 'section_text', 'section_id_gflm_section', 'gflm_section']]

        self.gflm_section_result = gflm_section_join

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
    print('Feature mining defatult workflow...')
    fm = FeatureMining()
    fm.load_ipod(full_set=False)
    fm.fit()
    fm.predict()
    fm.section_features()
    print(fm.gflm.gflm_section.head(10))
    print(fm.gflm.gflm_word.head(10))
    print(fm.gflm_section_result.sort_values(by=['gflm_section'], ascending=False)[['feature', 'section_text']].head(20))
    print(fm.gflm_word_result.sort_values(by=['gflm_word'], ascending=False)[['feature', 'section_text']].head(20))
    print('Done.')
