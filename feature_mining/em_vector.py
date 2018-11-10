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

        # TODO: Remove these after testing validated
        # Parameters for temporary import transformation
        self.reviews_matrix = np.array([])
        self.pi_matrix = np.array([])
        self.topic_model_matrix = ()
        self.reviews_binary = np.array([])
        self.hidden_parameters_one_sentence_for_testing = {}
        self.hidden_parameters_background_one_sentence_for_testing = {}
        self.expose_sentence_sum_for_testing = None

    def import_data(self):
        print(type(self).__name__, '- import data...')
        self.import_data_temporary() # TODO: deactivate this when our data is available

    def import_data_temporary(self):
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
        for aspect in self.topic_model:
            na += 1

        words_dict = {}
        for aspect in self.topic_model.keys():
            #print(aspect, len(self.topic_model[aspect]))
            for word in self.topic_model[aspect]:
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

        # construct the aspect map
        current_aspect = 0
        aspects_map = {}
        for one_aspect in sorted(self.pi[0][0].keys()):
            aspects_map[one_aspect] = current_aspect
            current_aspect += 1

        # initialize pi
        # pi_matrix = np.random.dirichlet(np.ones(m), na).transpose()
        pi_matrix = np.zeros(m * na).reshape(m, na)
        section_id = 0
        pi_matrix = np.zeros(m * na).reshape(m, na)
        for reviewNum in range(0, len(self.reviews)):
            for lineNum in range(0, len(self.reviews[reviewNum])):
                for aspect in self.pi[reviewNum][lineNum]:
                    pi_matrix[section_id][aspects_map[aspect]] = self.pi[reviewNum][lineNum][aspect]
                section_id += 1

        # initialize topic model with zeros
        topic_model_matrix = np.zeros(nw * na).reshape(nw, na)
        for aspect in self.topic_model:
            for word in self.topic_model[aspect]:
                topic_model_matrix[words_map[word]][aspects_map[aspect]] = self.topic_model[aspect][word]

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
        self.aspects_map = aspects_map
        self.words_map = words_map
        self.words_list = word_list
        self.background_probability = background_probability_vector

    def initialize_parameters(self):
        print(type(self).__name__, '- initialize parameters...')

        # Compute binary reviews matrix (1 if word in sentence, 0 if not) (same dimensions as reviews)
        self.reviews_binary = np.where(self.reviews_matrix > 0, 1, 0)

        # Compute sparse matrix
        self.reviews_matrix = csr_matrix(self.reviews_matrix)
        self.reviews_binary = csr_matrix(self.reviews_binary)

        # Initialize hidden_parameters
        self.hidden_parameters = []
        hidden_parameters_one_sentence = np.zeros(self.m * self.v).reshape(self.m, self.v)
        for sentence in range(0, self.m):
            self.hidden_parameters.append(hidden_parameters_one_sentence)

        # Initialize hidden_parameters_background
        self.hidden_parameters_background = csr_matrix((self.m, self.v))

        # TODO: enable this code when ready
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
            print(30 * '*', "Start sentence:", sentence, ' ', 30 * '*')

            # Compute topic model for sentence as review_binary[sentence_s]^T * topic_model
            # TODO: extract this initialization in initialize_parameters
            topic_model_sentence = self.reviews_binary[sentence].reshape(self.v, 1).multiply(self.topic_model_matrix)

            # Compute sum of review * topic_model for sentence_s
            sentence_sum = topic_model_sentence.dot(self.pi_matrix[sentence])

            # We will have 0 values for sentence_sum, for missing words; to avoid division by 0, sentence_sum(word) = 1
            sentence_sum = np.where(sentence_sum == 0, 1, sentence_sum)

            # Compute hidden_parameters for sentence_s
            # TODO: not optimal (transpose twice). Redo please.
            hidden_parameters_sentence = (
                        (topic_model_sentence.multiply(self.pi_matrix[sentence])).T / sentence_sum).T

            # TODO: Compute hidden_parameters_background
            background_probability = self.lambda_background * \
                                     self.reviews_binary[sentence].T.multiply(
                                         self.background_probability.reshape(self.v, 1))

            # TODO: Optimize this, test sparse implementation
            hidden_parameters_background_sentence = background_probability / \
                                                    (background_probability +
                                                     ((1 - self.lambda_background) * sentence_sum).reshape(self.v, 1))

            # Only used for testing during implementation. To be deleted.
            self.hidden_parameters_one_sentence_for_testing = hidden_parameters_sentence
            self.expose_sentence_sum_for_testing = sentence_sum
            self.hidden_parameters_background_one_sentence_for_testing = hidden_parameters_background_sentence
            print(30 * '*', "End sentence:", sentence, ' ', 30 * '*')


if __name__ == '__main__':
    em = ExpectationMaximizationVector()
    em.em()

"""
    * Notation:
            v = number of words in vocabulary
            m  = number of sentences (lines) in all reviews
            na = number of aspects
    
    * Review matrix:
            Sentence/Word | word 1 ... ... ... ... word v
            ---------------------------------------------------
            Sentence 1    | count(s_1,w_1) ... ...  count(s_1, w_v)
            Sentence 2    | count(s_2,w_2) ... ...  count(s_2, w_v)
            ...    ...     ... ...     ...     ...     ...
            Sentence m    | count(s_m, w_1)... ...  count(s_m, w_v)
    
    * Topic model
            Word/Aspect    | aspect 1   ...     ...     aspect na
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
            Sentence/Aspect | aspect 1 ...      ...    aspect na
            -----------------------------------------------------
            Sentence 1      | pi(s1,a1)  ...   ...     pi(s1, a_na)
            ...        ...             ....            ...     ...
            Sentence m      | pi(sm,a1)  ...   ...     pi(sm, a_na)
    
    * Hidden parameters (list of one sparse matrix for each sentence)
        - [0]:
            Word / Aspect | aspect 1 ... ... ... ... aspect na
            ---------------------------------------------------
            word 1        | 0.0         ...         ...   0.0
            word 2        | 0.0         ...         ...   0.0
            ...    ...    |         ...     ...     ...
            word v       | 0.0    ...              ...   0.0
            
        - [...]:
            ... ... ... 
        - [m]: 
            Word / Aspect | aspect 1 ... ... ... ... aspect na
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