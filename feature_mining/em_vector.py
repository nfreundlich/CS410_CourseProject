import numpy as np
from scipy.sparse import csr_matrix
from feature_mining.em_base import ExpectationMaximization


class ExpectationMaximizationVector(ExpectationMaximization):
    """
    Vectorized implementation of EM algorithm.
    """

    def __init__(self, dump_path="../tests/data/em_01/"):
        print(type(self).__name__, '- init...')
        ExpectationMaximization.__init__(self, dump_path=dump_path)

        # Parameters for matrix result interpretation
        self.aspects_map = {}
        self.words_map = {}
        self.words_list = {}

        # TODO: one more optimization place
        self.topic_model_sentence_matrix = None

        # TODO: Remove these after testing validated
        # Parameters for temporary import transformation
        self.reviews_matrix = np.array([])
        self.pi_matrix = np.array([])
        self.topic_model_matrix = ()
        self.reviews_binary = np.array([])
        self.previous_pi_matrix = None
        self.hidden_parameters_one_sentence_for_testing = {}
        self.hidden_parameters_background_one_sentence_for_testing = {}
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
        self.import_data_temporary() # TODO: deactivate this when our data is available

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
            #print(feature, len(self.topic_model[feature]))
            for word in self.topic_model[feature]:
                words_dict[word] = True
        nw = len(words_dict.keys()) # 7266
        word_list = sorted(words_dict.keys())
        words_map = {}
        for word_id in range(0, len(word_list)):
            words_map[word_list[word_id]] = word_id

        #print("m", "v", "na")
        #print(m, v, na)

        # initialize reviews with zeros
        reviews_matrix = np.zeros(m * nw).reshape(m, nw)

        # construct the review matrix with count values for each words
        section_id = 0
        for reviewNum in range(0, len(self.reviews)):
            for lineNum in range(0, len(self.reviews[reviewNum])):
                for word in self.reviews[reviewNum][lineNum]:
                    reviews_matrix[section_id][words_map[word]] = self.reviews[reviewNum][lineNum][word]
                section_id += 1

        # check first line
        #for i in range(0, len(reviews_matrix[0])):
        #    if reviews_matrix[0][i] != 0:
        #        print(i, reviews_matrix[0][i], word_list[i])

        # construct the feature map
        current_feature = 0
        features_map = {}
        for one_feature in sorted(self.pi[0][0].keys()):
            features_map[one_feature] = current_feature
            current_feature += 1

        # initialize pi
        # pi_matrix = np.random.dirichlet(np.ones(m), na).transpose()
        pi_matrix = np.zeros(m * na).reshape(m, na)
        section_id = 0
        pi_matrix = np.zeros(m * na).reshape(m, na)
        for reviewNum in range(0, len(self.reviews)):
            for lineNum in range(0, len(self.reviews[reviewNum])):
                for feature in self.pi[reviewNum][lineNum]:
                    pi_matrix[section_id][features_map[feature]] = self.pi[reviewNum][lineNum][feature]
                section_id += 1

        # initialize topic model with zeros
        # TODO: sparse this!
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
        self.iv = csr_matrix(np.ones(self.v)).reshape(self.v, 1)
        self.features_map = features_map
        self.words_map = words_map
        self.words_list = word_list
        self.background_probability = background_probability_vector

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
        self.reviews_binary = np.where(self.reviews_matrix > 0, 1, 0)

        # Compute sparse matrix
        self.reviews_matrix = csr_matrix(self.reviews_matrix)
        self.reviews_binary = csr_matrix(self.reviews_binary)

        # Initialize hidden_parameters
        self.hidden_parameters = []
        hidden_parameters_one_sentence = np.zeros(self.f * self.v).reshape(self.f, self.v)
        for sentence in range(0, self.m):
            self.hidden_parameters.append(hidden_parameters_one_sentence)

        # Initialize hidden_parameters_background
        self.hidden_parameters_background = csr_matrix((self.m, self.v))

        # Compute topic model for sentence as review_binary[sentence_s]^T * topic_model
        # TODO: extract this initialization in initialize_parameters and OPTIMIZE, OPTIMIZE
        self.topic_model_sentence_matrix = []#((self.m, self.v, self.f))
        for sentence in range(0, self.m):
            self.topic_model_sentence_matrix.append(csr_matrix(
                self.reviews_binary[sentence].reshape(self.v, 1).multiply(self.topic_model_matrix)))

        # TODO: enable this code when ready to generate random pi-s
        if False:
            self.pi_matrix = np.random.dirichlet(np.ones(self.m), self.f).transpose()

    def e_step(self):
        """
         Vectorized e-step.

         :return:
        """
        print(type(self).__name__, '- e_step...')

        # TODO: after testing, change 1 to m (to treat all sentences)
        for sentence in range(0, 1):  # TODO: replace with 'm' (now not needed)
            print(30 * '*', "E-Step Start sentence:", sentence, ' ', 30 * '*')

            # Compute sum of review * topic_model for sentence_s
            sentence_sum = (self.topic_model_sentence_matrix[sentence].dot(self.pi_matrix[sentence]))

            # We will have 0 values for sentence_sum, for missing words; to avoid division by 0, sentence_sum(word) = 1
            sentence_sum = np.where(sentence_sum == 0, 1, sentence_sum)

            # Compute hidden_parameters for sentence_s
            # TODO: not optimal (transpose twice). Redo please.
            hidden_parameters_sentence = (
                        (self.topic_model_sentence_matrix[sentence].multiply(self.pi_matrix[sentence])).T / sentence_sum).T

            # Compute hidden_parameters_background
            background_probability = csr_matrix(self.lambda_background * \
                                     self.reviews_binary[sentence].T.multiply(
                                         self.background_probability.reshape(self.v, 1)))

            hidden_parameters_background_sentence = csr_matrix(background_probability / \
                                                    (background_probability +
                                                     ((1 - self.lambda_background) * sentence_sum).reshape(self.v, 1)))

            if True: # Used only for testing
                # Only used for testing during implementation. To be deleted.
                self.hidden_parameters_one_sentence_for_testing = hidden_parameters_sentence
                self.expose_sentence_sum_for_testing = sentence_sum
                self.hidden_parameters_background_one_sentence_for_testing = hidden_parameters_background_sentence.todense()

                # this is to be kept
                self.hidden_parameters[sentence] = hidden_parameters_sentence.copy()
                self.hidden_parameters_background[sentence] = hidden_parameters_background_sentence.T.copy()

                # print(30 * '*', "End sentence:", sentence, ' ', 30 * '*')

    def m_step(self):
        """
                 Vectorized e-step.

                 :return:
                """
        print(type(self).__name__, '- m_step...')
        self.previous_pi_matrix = self.pi_matrix.copy()

        # TODO: after testing, change 1 to m (to treat all sentences)
        for sentence in range(0, 1):  # TODO: replace with 'm' (now not needed)
            print(30 * '*', "Start sentence:", sentence, ' ', 30 * '*')
            self.m_sum = csr_matrix(self.iv - self.hidden_parameters_background_one_sentence_for_testing).T \
                            .multiply(self.reviews_matrix[sentence]) \
                            .dot(self.hidden_parameters_one_sentence_for_testing)
            self.denom = self.m_sum.sum()

            # TODO: delete this useless line after testing. All we need is m_sum.
            self.nom = self.m_sum.item(0)

            # update pi for a sentence
            self.pi_matrix[sentence] = self.m_sum / self.m_sum.sum()

            print(30 * '*', "End sentence:", sentence, ' ', 30 * '*')

    def compute_cost(self):
        print(type(self).__name__, '- compute cost...')
        delta = self.pi_matrix - self.previous_pi_matrix

        return 0.0


if __name__ == '__main__':
    em = ExpectationMaximizationVector()
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