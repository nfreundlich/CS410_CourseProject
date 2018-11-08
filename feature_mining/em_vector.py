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
        #self.reviews = np.load(self.dump_path + "Reviews.npy")
        #self.topic_model = np.load(self.dump_path + 'TopicModel.npy').item()
        #self.background_probability = np.load(self.dump_path + 'BackgroundProbability.npy').item()

    def initialize_parameters(self):
        print(type(self).__name__, '- initialize parameters...')
        #self.hidden_parameters = np.load(self.dump_path + "HP.npy")
        #self.hidden_parameters_background = np.load(self.dump_path + "HPB.npy")
        #self.pi = np.load(self.dump_path + "PI.npy")


if __name__ == '__main__':
    em = ExpectationMaximizationVector()
    em.em()

