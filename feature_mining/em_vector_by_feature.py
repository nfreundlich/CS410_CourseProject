import numpy as np
from scipy.sparse import csr_matrix
from feature_mining.em_base import ExpectationMaximization
from feature_mining import ParseAndModel
from datetime import datetime
import os
import logging


class EmVectorByFeature(ExpectationMaximization):
    """
    Vectorized implementation of EM algorithm. Loops over features.
    """

    def __init__(self,  explicit_model: ParseAndModel = None,
                 lambda_background: float = 0.7, max_iter: int = 50, delta_threshold: float = 1e-6):
        """
        Constructor for EM class (looping by feature)

        :param explicit_model: ParseAndModel object containing topic models, background model and word counts
        :param lambda_background: Assumed P( background model) - probabilitiy that a word comes from the background distribution
        :param max_iter: Maximum number of EM iterations to run if delta threshold is not reached
        :param delta_threshold: Delta Pi threshold at which the EM algorithm will terminate
        """

        logging.info(type(self).__name__, '- init...')
        ExpectationMaximization.__init__(self, dump_path=None)

        # User set parameters
        self.max_iter = max_iter
        self.lambda_background = lambda_background
        self.delta_threshold = delta_threshold

        self.pi_matrix = np.array([])
        self.previous_pi_matrix = None
        self.reviews_binary = np.array([])
        self.hidden_parameters_background_estep = None

        self.explicit_model = None
        self.m = None
        self.v = None
        self.k = None

        self.reviews_matrix = None
        self.topic_model = None
        self.background_probability = None

        # if explicit model exists initialize class variables, else skip
        if explicit_model is not None:
            self.explicit_model = explicit_model

            # Parameters related to collection size
            self.m = explicit_model.model_results["section_word_counts_matrix"].shape[0]
            self.v = explicit_model.model_results["section_word_counts_matrix"].shape[1]
            self.k = explicit_model.model_results["model_feature_matrix"].shape[1]

            # Parameters computed from collection
            self.reviews_matrix = explicit_model.model_results["section_word_counts_matrix"]
            self.topic_model = explicit_model.model_results["model_feature_matrix"]
            self.background_probability = explicit_model.model_results["model_background_matrix"]

            logging.info("Explicit models have been imported into EM")

        else:
            logging.warning("Parse and model output was not included as an argument and will need to be set manually")

    def import_data(self, explicit_model: ParseAndModel = None):
        """
        Takes a ParseAndModel object and saves the data for EM algorithm use

        :param explicit_model: A ParseAndModel object containing the necessary information for EM calculations
        :return: None
        """
        self.explicit_model = explicit_model

        self.m = explicit_model.model_results["section_word_counts_matrix"].shape[0]
        self.v = explicit_model.model_results["section_word_counts_matrix"].shape[1]
        self.k = explicit_model.model_results["model_feature_matrix"].shape[1]

        # Parameters computed from collection
        self.reviews_matrix = explicit_model.model_results["section_word_counts_matrix"]
        self.topic_model = explicit_model.model_results["model_feature_matrix"]
        self.background_probability = explicit_model.model_results["model_background_matrix"]

        logging.info("Explicit model has been imported - algorithm can be started")

    def initialize_parameters(self):
        """
        Initialize helper parameters for E-M.

        :return: None
        """
        logging.info(type(self).__name__, '- initialize parameters...')

        # Compute binary reviews matrix (1 if word in sentence, 0 if not) (same dimensions as reviews)
        self.reviews_binary = self.reviews_matrix.sign()

        # Compute sparse matrix
        # self.reviews_matrix = csr_matrix(self.reviews_matrix)
        # self.reviews_binary = csr_matrix(self.reviews_binary)

        # Initialize hidden_parameters
        self.hidden_parameters = []
        hidden_parameters_one_feature = csr_matrix((self.m, self.v))
        for feature in range(0, self.k):
            self.hidden_parameters.append(hidden_parameters_one_feature)

        # Initialize hidden_parameters_background
        self.hidden_parameters_background = csr_matrix((self.m, self.v))

        # Calculate E-step background hidden parameter numerator - this won't change throughout the process
        self.hidden_parameters_background_estep = self.reviews_binary.multiply(self.lambda_background).multiply(
            self.background_probability)

        # Generates equal weight initialization for testing
        # if False:
        #    self.pi_matrix = np.full((self.m, self.f), 1/self.f)

        # Generates random intitialization
        if True:
            self.pi_matrix = np.random.dirichlet(np.ones(self.k), self.m)

    def e_step(self):
        """
         Vectorized e-step.

         Estimates value of hidden parameters based on current pi parameters

         :return: None
        """
        logging.info(type(self).__name__, '- e_step...')

        hidden_param_sum = 0
        for feature in range(0, self.k):
            logging.info(30 * '*', "E-Step Start feature:", feature, ' ', 30 * '*')

            pi_topic = np.dot(self.pi_matrix[:, feature, np.newaxis], self.topic_model[np.newaxis, :, feature])

            self.hidden_parameters[feature] = self.reviews_binary.multiply(pi_topic)

            if feature == 0:
                # First loop, initialze sum
                hidden_param_sum = self.hidden_parameters[feature]
            else:
                # Not first loop, add to existing sum
                hidden_param_sum += self.hidden_parameters[feature]

        # Calculate denominator for background hidden parameters
        hidden_background_denom = self.hidden_parameters_background_estep + hidden_param_sum.multiply(
            1 - self.lambda_background)

        # Normalize hidden background parameters
        self.hidden_parameters_background = self.hidden_parameters_background_estep.multiply(
            hidden_background_denom.power(-1))

        # take element-wise sum hidden feature params ^ -1 so we can divide instead of multiplying
        hidden_param_sum = hidden_param_sum.power(-1)

        # Normalize hidden parameters for each feature
        for feature in range(0, self.k):
            logging.info(30 * '*', "E-Step Start feature normalization:", feature, ' ', 30 * '*')

            self.hidden_parameters[feature] = self.hidden_parameters[feature].multiply(hidden_param_sum)

    def m_step(self):
        """
        Vectorized m-step.

        Re-estimates parameters based on current hidden parameter values

         :return: None
        """
        logging.info(type(self).__name__, '- m_step...')

        self.previous_pi_matrix = self.pi_matrix.copy()

        # Calculate non-feature dependent portion of m-step numerator: c(w,s) (1-P(z_sw = B))
        multiplier = self.hidden_parameters_background.multiply(-1)
        multiplier.data += 1
        multiplier = self.reviews_matrix.multiply(multiplier)

        # For each feature calculate numerator
        pi_sums = 0
        for feature in range(0, self.k):
            logging.info(30 * '*', "M-step start pi calculation:", feature, ' ', 30 * '*')

            new_pi = multiplier.multiply(self.hidden_parameters[feature]).sum(axis=1)

            # keep running total for denominator
            if feature == 0:
                # if first feature initialize
                self.pi_matrix = new_pi
                pi_sums = new_pi
            else:
                # if not first feature then add on to pi matrix
                self.pi_matrix = np.column_stack((self.pi_matrix, new_pi))
                pi_sums += new_pi

        # Pi sums is a dense matrix so we replace 0s with 1s
        pi_sums = np.where(pi_sums == 0, 1, pi_sums)

        self.pi_matrix = np.multiply(self.pi_matrix, np.power(pi_sums, -1))

    def compute_cost(self):
        """
        Calculates the amount of change between the last iteration's pi parameter values and the current iteration's pi
        parameter values for comparison to the delta threshold.

        Total change is calculated as the sum of the squared differences between the two sets of parameters.

        :return:
        """
        logging.info(type(self).__name__, '- compute cost...')

        delta = np.square(np.subtract(self.pi_matrix, self.previous_pi_matrix))

        return delta.sum()


if __name__ == '__main__':
    em = EmVectorByFeature()
    em.em()

"""
    * Notation:
            v = number of words in vocabulary
            m  = number of sections (lines) across all documents
            k = number of features
            
            Note: Python is zero indexed, but for simplicity of explanation we use 1-indexing here
    
    * Section word counts matrix (Sparse)
            Section/Word | word 1 ... ... ... ... word v
            ---------------------------------------------------
            Section 1    | count(s_1,w_1) ... ...  count(s_1, w_v)
            Section 2    | count(s_2,w_2) ... ...  count(s_2, w_v)
            ...    ...     ... ...     ...     ...     ...
            Section m    | count(s_m, w_1)... ...  count(s_m, w_v)
    
    * Topic model (Dense)
            Word/feature    | feature 1 ...     ...     feature k
            -----------------------------------------------------
            word 1         | p(w1 | f1) ...     ...     p(w1 | fk)
            ...        ...              ...            ...     ...
            word v         | p(wv | fk) ...     ...     p(wv | fk)
    
    * Background probability (Dense)
            Word           | Background probability
            ----------------------------------------
            word 1         |   p(w1 | B)
            ...        ... |   ...
            word v         |   p(wv | B)
    
    * PI (Dense)
            Section/feature | feature 1 ...      ...    feature k
            -----------------------------------------------------
            Section 1       | pi(s1,f1)  ...   ...     pi(s1, fk)
            ...        ...             ....            ...     ...
            Section m       | pi(sm,f1)  ...   ...     pi(sm, fk)
    
    * Hidden parameters (list of one sparse matrix for each feature)
        - [1]:
            section / word    | word 1 ... ... ... ... word v
            --------------------------------------------------------
            section 1         | p(z 1, 1 = 1)  ...     p(z 1, v = 1)
            section 2         | p(z 2, 1 = 1)  ...     p(z 2, v = 1)
            ...    ...        |         ...    ...     ...
            section m         | p(z m, 1 = 1)  ...     p(z m, v = 1)
            
        - [...]:
            ... ... ... 
        - [k]: 
            section / word    | word 1 ... ... ... ... word v
            --------------------------------------------------------
            section 1         | p(z 1, 1 = k)  ...     p(z 1, v = k)
            section 2         | p(z 2, 1 = k)  ...     p(z 2, v = k)
            ...    ...        |         ...    ...     ...
            section m         | p(z m, 1 = k)  ...     p(z m, v = k)

    * Hidden parameters background (sparse)
            section / word    | word 1 ... ... ... ...  word v
            --------------------------------------------------------
            section 1         | p(z 1, 1 = B)  ...     p(z 1, v = B)
            section 2         | p(z 2, 1 = B)  ...     p(z 2, v = B)
            ...    ...        |         ...    ...     ...
            section m         | p(z m, 1 = B)  ...     p(z m, v = B)
"""
