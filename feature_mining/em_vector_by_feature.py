import numpy as np
from scipy.sparse import csr_matrix
from feature_mining.em_base import ExpectationMaximization
from feature_mining import parse_and_model
from feature_mining import ParseAndModel
from datetime import datetime
import os

class EmVectorByFeature(ExpectationMaximization):
    """
    Vectorized implementation of EM algorithm.
    """

    def __init__(self, dump_path="../tests/data/em_01/"):
        print(type(self).__name__, '- init...')
        ExpectationMaximization.__init__(self, dump_path=dump_path)

        # Parameters for matrix result interpretation
        self.features_map = {}
        self.words_map = {}
        self.words_list = {}

        # TODO: one more optimization place
        self.topic_model_sentence_matrix = None
        self.iv = None  # identity array of v size

        # TODO: Remove these after testing validated
        # Parameters for temporary import transformation
        self.reviews_matrix = np.array([])
        self.pi_matrix = np.array([])
        self.topic_model_matrix = ()
        self.reviews_binary = np.array([])
        self.previous_pi_matrix = None
        self.expose_sentence_sum_for_testing = None
        self.denom = 0.0
        self.nom = 0.0
        self.m_sum = None

    def import_data(self):
        """
        Needed data structures could be further transformed here.

        Input:
            please see import_data_temporary (for now)
        Output:
            - self.reviews_matrix: matrix representation of Reviews
            - self.topic_model_matrix: matrix representation of TopicModel
            - self.pi_matrix: matrix representation of PI
            - self.m: number of sections (sentences)
            - self.v: number of words in vocabulary
            - self.f: number of features/aspects
            - self.iv: identity vector of b length
            - self.features_map: used for testing - maps features to id-s
            - self.words_map: used for testing - maps words to id-s
            - self.words_list: list of words (inverse of words_map)
            - self.background_probability = background_probability_vector
        :return:
        """
        print(type(self).__name__, '- import data...')

        # TODO: determine how best to pass in necessary arguments

        print(os.getcwd())

        pm_inst = ParseAndModel(feature_list=["sound", "battery", ["screen", "display"]], filename='demo_files/iPod.final', nlines=5)

        print("formatting feature list")
        #feature_list = pm_inst.format_feature_list(feature_list=["sound", "battery", ["screen", "display"]])

        print("reading annotated data")
        #annotated_data = pm_inst.read_annotated_data(filename='demo_files/iPod.final', nlines=None)

        print("parsing text and building explicit feature models: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        #em_input = pm_inst.build_explicit_models(text_set=annotated_data["section_list"], feature_set=feature_list)
        print("parsing text and building explicit feature models: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        # Parameters related to collection size
        # self.m = em_input["section_word_counts_matrix"].shape[0]
        # self.v = em_input["section_word_counts_matrix"].shape[1]
        # self.f = em_input["model_feature_matrix"].shape[1]
        self.m = pm_inst.model_results["section_word_counts_matrix"].shape[0]
        self.v = pm_inst.model_results["section_word_counts_matrix"].shape[1]
        self.f = pm_inst.model_results["model_feature_matrix"].shape[1]

        # Parameters computed from collection
        # self.reviews_matrix = em_input["section_word_counts_matrix"]
        # self.topic_model = em_input["model_feature_matrix"].toarray() # TODO remove toarray after fixing parse and model
        # self.background_probability = em_input["model_background_matrix"]
        self.reviews_matrix = pm_inst.model_results["section_word_counts_matrix"]
        self.topic_model = pm_inst.model_results["model_feature_matrix"].toarray() # TODO remove toarray after fixing parse and model
        self.background_probability = pm_inst.model_results["model_background_matrix"]


    def import_data_temporary(self):
        """
        Transforms data read from a snapshot of Santu's data, after 1 execution of E-M steps.
        Input:
        - Reviews.npy: np dump of the Reviews structure
        - TopicModel.npy: np dump of TopicModel
        - BackgroundProbability: np dump of BackgroundProbability
        - HP: np dump of hidden parameters
        - PI: np dump of pi (important for testing purposes, since this is randomly generated)
        :return:
        """
        # TODO: this should be deleted once the implementation is safe
        print(type(self).__name__, '- import data ********temporary********...')
        self.reviews = np.load(self.dump_path + "Reviews.npy")
        self.topic_model = np.load(self.dump_path + 'TopicModel.npy').item()
        self.background_probability = np.load(self.dump_path + 'BackgroundProbability.npy').item()
        self.hidden_parameters = np.load(self.dump_path + "HP.npy")
        self.hidden_parameters_background = np.load(self.dump_path + "HPB.npy")
        self.pi = np.load(self.dump_path + "PI.npy")

        """
        Prepare data for testing vectorised solution.
        Want to convert the photo of Santu's data for vectorised needs.
        :return:
        """
        m = 0   # number of sentences (lines) in all reviews - 8799
        nw = 0  # number of words in vocabulary - 7266
        na = 0  # number of aspects - 9

        for reviewNum in range(0, len(self.reviews)):
            for lineNum in range(0, len(self.reviews[reviewNum])):
                m += 1
        for feature in self.topic_model:
            na += 1

        words_dict = {}
        for feature in self.topic_model.keys():
            for word in self.topic_model[feature]:
                words_dict[word] = True
        nw = len(words_dict.keys()) # 7266
        word_list = sorted(words_dict.keys())
        words_map = {}
        for word_id in range(0, len(word_list)):
            words_map[word_list[word_id]] = word_id

        # initialize reviews with zeros
        reviews_matrix = np.zeros(m * nw).reshape(m, nw)

        # construct the review matrix with count values for each words
        section_id = 0
        for reviewNum in range(0, len(self.reviews)):
            for lineNum in range(0, len(self.reviews[reviewNum])):
                for word in self.reviews[reviewNum][lineNum]:
                    reviews_matrix[section_id][words_map[word]] = self.reviews[reviewNum][lineNum][word]
                section_id += 1

        # construct the feature map
        current_feature = 0
        features_map = {}
        for one_feature in sorted(self.pi[0][0].keys()):
            features_map[one_feature] = current_feature
            current_feature += 1

        # initialize pi
        section_id = 0
        pi_matrix = np.zeros(m * na).reshape(m, na)
        for reviewNum in range(0, len(self.reviews)):
            for lineNum in range(0, len(self.reviews[reviewNum])):
                for feature in self.pi[reviewNum][lineNum]:
                    pi_matrix[section_id][features_map[feature]] = self.pi[reviewNum][lineNum][feature]
                section_id += 1

        # initialize topic model with zeros
        topic_model_matrix = np.zeros(nw * na).reshape(nw, na)
        for feature in self.topic_model:
            for word in self.topic_model[feature]:
                topic_model_matrix[words_map[word]][features_map[feature]] = self.topic_model[feature][word]

        # initialize hidden parameters for background
        self.words_map = words_map
        background_probability_vector = np.zeros(nw).reshape(nw, 1)
        for k, v in self.background_probability.items():
            background_probability_vector[self.words_map[k]] = v
        background_probability_vector = background_probability_vector.squeeze()

        # update class parameters with matrices
        # TODO: clean this up to use only one set of input data
        self.reviews_matrix = reviews_matrix
        self.topic_model_matrix = topic_model_matrix
        self.pi_matrix = pi_matrix
        self.m = m
        self.v = nw
        self.f = na
        self.iv = np.ones(self.v).reshape(self.v, 1).T
        self.features_map = features_map
        self.words_map = words_map
        self.words_list = word_list
        self.background_probability = background_probability_vector.reshape(self.v, 1)

    def initialize_parameters(self):
        """
        Initialize helper parameters for E-M.

        :return:
            - reviews_binary: binary matrix of reviews_matrix (1 wherever count>0)
            - hidden_parameters: 0-matrix of hidden parameters
            - hidden_parameters_background: 0-matrix of h.p. background
            - pi_matrix: random dirichlet initialization (DEACTIVATED MOMENTARILY)
        """
        print(type(self).__name__, '- initialize parameters...')

        # Compute binary reviews matrix (1 if word in sentence, 0 if not) (same dimensions as reviews)
        self.reviews_binary = self.reviews_matrix.sign()

        # Compute sparse matrix
        self.reviews_matrix = csr_matrix(self.reviews_matrix)
        self.reviews_binary = csr_matrix(self.reviews_binary)

        # Initialize hidden_parameters
        self.hidden_parameters = []
        hidden_parameters_one_feature = csr_matrix((self.m, self.v))
        for feature in range(0, self.f):
            self.hidden_parameters.append(hidden_parameters_one_feature)

        # Initialize hidden_parameters_background
        self.hidden_parameters_background = csr_matrix((self.m, self.v))

        # Calculate E-step background hidden parameter numerator - this won't change throughout the process
        self.hidden_parameters_background_estep = self.reviews_binary.multiply(self.lambda_background).multiply(self.background_probability)

        # Compute topic model for sentence as review_binary[sentence_s]^T * topic_model
        # TODO: extract this initialization in initialize_parameters and OPTIMIZE, OPTIMIZE
        # self.topic_model_sentence_matrix = []#((self.m, self.v, self.f))
        # for sentence in range(0, self.m):
        #    self.topic_model_sentence_matrix.append(csr_matrix(
        #        self.reviews_binary[sentence].reshape(self.v, 1).multiply(self.topic_model_matrix)))

        # TODO disable when testing is done - currently starting with even weights
        if False:
            self.pi_matrix = np.full((self.m, self.f), 1/self.f)

        # TODO: enable this code when ready to generate random pi-s
        if True:
            self.pi_matrix = np.random.dirichlet(np.ones(self.m), self.f).transpose()

    def e_step(self):
        """
         Vectorized e-step.

         :return:
        """
        print(type(self).__name__, '- e_step...')

        # TODO: after testing, change 1 to k (to treat all features)
        for feature in range(0, self.f):  # TODO: replace with 'f' (now not needed)
            print(30 * '*', "E-Step Start feature:", feature, ' ', 30 * '*')

            pi_topic = np.dot(self.pi_matrix[:,feature,np.newaxis], self.topic_model[np.newaxis, :, feature])

            self.hidden_parameters[feature] = self.reviews_binary.multiply(pi_topic)

            if feature==0:
                # First loop, initialze sum
                hidden_param_sum = self.hidden_parameters[feature]
            else:
                # Not first loop, add to existing sum
                hidden_param_sum += self.hidden_parameters[feature]

        # Calculate denominator for background hidden parameters
        hidden_background_denom = self.hidden_parameters_background_estep + hidden_param_sum.multiply(1-self.lambda_background)

        # Normalize hidden background parameters
        self.hidden_parameters_background = self.hidden_parameters_background_estep.multiply(hidden_background_denom.power(-1))

        # take element-wise sum hidden feature params ^ -1 so we can divide instead of multiplying
        hidden_param_sum = hidden_param_sum.power(-1)

        # Normalize hidden parameters for each feature
        for feature in range(0, self.f):
            print(30 * '*', "E-Step Start feature normalization:", feature, ' ', 30 * '*')

            self.hidden_parameters[feature] = self.hidden_parameters[feature].multiply(hidden_param_sum)



    def m_step(self):
        """
                 Vectorized e-step.

                 :return:
                """
        print(type(self).__name__, '- m_step...')

        self.previous_pi_matrix = self.pi_matrix.copy()

        # Calculate non-feature dependent portion of m-step numerator: c(w,s) (1-P(z_sw = B))
        multiplier = self.hidden_parameters_background.multiply(-1)
        multiplier.data -= 1
        multiplier = self.reviews_matrix.multiply(multiplier)

        # For each feature calculate numerator
        for feature in range(0, self.f):
            print(30 * '*', "M-step start pi calculation:", feature, ' ', 30 * '*')

            new_pi = multiplier.multiply(self.hidden_parameters[feature]).sum(axis=1)

            # keep running total for denominator
            if feature==0:
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
        print(type(self).__name__, '- compute cost...')
        # TODO: fix distance formula
        delta = np.subtract(self.pi_matrix, self.previous_pi_matrix)

        return np.max(np.positive(delta))


if __name__ == '__main__':
    em = EmVectorByFeature()
    em.em()

"""
    * Notation:
            v = number of words in vocabulary
            m  = number of sentences (lines) in all reviews
            f = number of features
    
    * Review matrix:
            Sentence/Word | word 1 ... ... ... ... word v
            ---------------------------------------------------
            Sentence 1    | count(s_1,w_1) ... ...  count(s_1, w_v)
            Sentence 2    | count(s_2,w_2) ... ...  count(s_2, w_v)
            ...    ...     ... ...     ...     ...     ...
            Sentence m    | count(s_m, w_1)... ...  count(s_m, w_v)
    
    * Topic model
            Word/feature    | feature 1   ...     ...     feature na
            -----------------------------------------------------
            word 1         | tm(w1,a1) ...      ...    tm(w1, a_na)
            ...        ...             ....            ...     ...
            word v        | tm(w_v, a_na) ... ....   tm(w_na, a_na)
    
    * Background probability
            Word           | Background probability
            ----------------------------------------
            word 1         |   bp_1
            ...        ... |   ...
            word v        |   bp_v
    
    * PI
            Sentence/feature | feature 1 ...      ...    feature na
            -----------------------------------------------------
            Sentence 1      | pi(s1,a1)  ...   ...     pi(s1, a_na)
            ...        ...             ....            ...     ...
            Sentence m      | pi(sm,a1)  ...   ...     pi(sm, a_na)
    
    * Hidden parameters (list of one sparse matrix for each sentence)
        - [0]:
            Word / feature | feature 1 ... ... ... ... feature f
            ---------------------------------------------------
            word 1        | 0.0         ...         ...   0.0
            word 2        | 0.0         ...         ...   0.0
            ...    ...    |         ...     ...     ...
            word v       | 0.0    ...              ...   0.0
            
        - [...]:
            ... ... ... 
        - [m]: 
            Word / feature | feature 1 ... ... ... ... feature f
            ---------------------------------------------------
            word 1        | 0.0         ...         ...   0.0
            word 2        | 0.0         ...         ...   0.0
            ...    ...    |         ...     ...     ...
            word v       | 0.0    ...              ...   0.0

    * Hidden parameters background
            Sentence/Word | word 1 ... ... ... ... word v
            ---------------------------------------------------
            Sentence 1    | 0.0        ...     ...   0.0
            Sentence 2    | 0.0    ...         ...   0.0
            ...    ...     ... ...     ...     ...   ...
            Sentence m    | 0.0 ...             ...  0.0
"""