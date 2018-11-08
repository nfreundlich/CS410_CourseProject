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

    def import_data(self):
        print(type(self).__name__, '- import data...')
        self.reviews = np.load(self.dump_path + "Reviews.npy")
        self.topic_model = np.load(self.dump_path + 'TopicModel.npy').item()
        self.background_probability = np.load(self.dump_path + 'BackgroundProbability.npy').item()
        # TODO: from em.py, adapt em_prepare_data_for_testing

    def initialize_parameters(self):
        print(type(self).__name__, '- initialize parameters...')
        # TODO: from em.py, adapt first part of em_e_step_sparse

    def e_step(self):
        print(type(self).__name__, '- e_step...')
        # TODO: from em.py, adapt second part of em_e_step_sparse


if __name__ == '__main__':
    em = ExpectationMaximizationVector()
    em.em()

