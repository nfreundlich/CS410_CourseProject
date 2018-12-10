import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd
from feature_mining.em_vector_by_feature import EmVectorByFeature
import logging


class GFLM:
    """
    Implementation of gflm-word and gflm-section
    """
    def __init__(self, section_threshold:float = 0.35, word_threshold:float = 0.35, em_results:EmVectorByFeature=None,
                 hidden_background=None, hidden_params=None, pi_matrix=None):
        """
        Constructor for GFLM.
        :param section_threshold: Threshold for section.
        :param word_threshold: Threshold for word.
        :param em_results: EM object with computed matrices
        :param hidden_background: (if em_results not provided) hidden background matrix (section_no x V)
        :param hidden_params: (if em_results not provided) list hidden params matrices (feat_no * (section_no x V))
        :param pi_matrix: EM computed PI matrix (section_no x feat_no)
        """

        # TODO: [nfr] check if this is needed in release workflow
        if em_results is not None:
            logging.info("Using EM results object for GFLM")
            self.em_results=em_results

            self.hidden_background = em_results.hidden_parameters_background
            self.hidden_params = em_results.hidden_parameters
            self.pi_matrix = em_results.pi_matrix
        else:
            self.hidden_background = hidden_background
            self.hidden_params = hidden_params
            self.pi_matrix = pi_matrix

        self.f = self.pi_matrix.shape[1]
        self.section_threshold = section_threshold
        self.word_threshold = word_threshold

        self.gflm_section = None        # initialize gflm_section here for persistence
        self.gflm_word = None           # initialize gflm_word here for persistence
        self.gflm_section_all = None    # initialize here for persistence
        self.gflm_word_all = None       # initialize here for persistence

    def calc_gflm_word(self, word_threshold:float = None):
        """
        Compute GFLM word.
        :param word_threshold: threshold to be used for gflm word
        :return: DataFrame holding mapping from section_id to implicit feature_id based on GFLM word
        """
        if word_threshold is None:
            word_threshold = self.word_threshold

        # Pre-calculate 1-P(z s,w = B)
        background_multiplier = self.hidden_background.multiply(-1)
        background_multiplier.data += 1

        # Loop through features
        for feature in range(0, self.f):
            #self.em_results.hidden_parameters_background
            #self.em_results.hidden_parameters[feature]

            # self.hidden_params
            # self.hidden_background

            gflm_vals = self.hidden_params[feature].multiply(background_multiplier)
            gflm_vals = pd.DataFrame(gflm_vals.max(axis=1).toarray())
            gflm_vals['section_id'] = gflm_vals.index.values
            gflm_vals['implicit_feature_id'] = feature
            gflm_vals.columns = ['gflm_word', 'section_id', 'implicit_feature_id']
            #
            if feature == 0:
                 self.gflm_word_all = gflm_vals
                 self.gflm_word = gflm_vals[gflm_vals.gflm_word >= word_threshold]
            else:
                self.gflm_word_all = pd.concat([self.gflm_word_all, gflm_vals])
                self.gflm_word = pd.concat([self.gflm_word,gflm_vals[gflm_vals.gflm_word >= word_threshold]])

        self.gflm_word = self.gflm_word.reset_index(drop=True)
        self.gflm_word_all = self.gflm_word_all.reset_index(drop = True)

        return self.gflm_word

    def calc_gflm_section(self, section_threshold:float = None):
        """
        Compute GFLM considering threshold at sentence.
        :param section_threshold: threshold to be used for gflm section
        :return: DataFrame holding mapping from section_id to implicit feature_id based on GFLM section
        """

        if section_threshold is None:
            section_threshold=self.section_threshold

        gflm_vals = pd.DataFrame(self.pi_matrix)
        gflm_vals["section_id"] = gflm_vals.index.values
        gflm_vals = pd.melt(gflm_vals, id_vars='section_id')
        gflm_vals.columns = ['section_id', 'implicit_feature_id', 'gflm_section']
        gflm_vals = gflm_vals[['gflm_section', 'section_id', 'implicit_feature_id']]
        gflm_vals["implicit_feature_id"] = pd.to_numeric(gflm_vals["implicit_feature_id"])

        self.gflm_section_all = gflm_vals
        self.gflm_section = gflm_vals[gflm_vals.gflm_section >= section_threshold]

        self.gflm_section = self.gflm_section.reset_index(drop=True)
        self.gflm_section_all = self.gflm_section_all.reset_index(drop=True)

        return self.gflm_section

if __name__ == '__main__':
    pi_matrix = np.array([[.1, .2, .3], [.4, .5, .6]])
    hidden_background = np.array([[.1, .2, .3], [.4, .5, .6]])
    hidden_params = dict()
    hidden_params[0] = np.array([[.1, .2, .3], [.4, .5, .6]])
    hidden_params[1] = np.array([[.2, .2, .3], [.4, .4, .6]])
    hidden_params[2] = np.array([[.8, .8, .8], [.6, .6, .6]])

    gflm = GFLM(hidden_params=hidden_params, hidden_background=hidden_background, pi_matrix=pi_matrix)
    gflm.calc_gflm_section()
    gflm.calc_gflm_word()

    print(gflm.gflm_word)
    print(gflm.gflm_section)
    print("Done")
