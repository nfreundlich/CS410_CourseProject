import numpy as np
from scipy.sparse import csr_matrix
from feature_mining.em_base import ExpectationMaximization


class ExpectationMinimizationOriginal(ExpectationMaximization):
    """
    Original EM Algorithm as developed by Santu.
    """

    def __init__(self, dump_path="../tests/data/em_01/"):
        print(type(self).__name__, '- init...')
        ExpectationMaximization.__init__(self, dump_path=dump_path)

    def import_data(self):
        print(type(self).__name__, '- import data...')
        self.reviews = np.load(self.dump_path + "Reviews.npy")
        self.topic_model = np.load(self.dump_path + 'TopicModel.npy').item()
        self.background_probability = np.load(self.dump_path + 'BackgroundProbability.npy').item()

    def initialize_parameters(self):
        print(type(self).__name__, '- initialize parameters...')
        self.hidden_parameters = np.load(self.dump_path + "HP.npy")
        self.hidden_parameters_background = np.load(self.dump_path + "HPB.npy")
        self.pi = np.load(self.dump_path + "PI.npy")

    def e_step(self):
        print(type(self).__name__, '- e_step...')
        """
        E-Step of EM algo, as implemented by ***Santu***.
        Compute HP and BHP.

        Input:
            reviews
            topic_model
            pi
            background_probability
            lambda_background
            hidden_parameters
            hidden_parameters_background
        
        Output:
            updated hidden_parameters
            updated hidden_parameters_background
        """
        for reviewNum in range(0, len(self.reviews)):
            for lineNum in range(0, len(self.reviews[reviewNum])):
                for word in self.reviews[reviewNum][lineNum]:
                    my_sum = 0
                    for aspect in self.topic_model:
                        my_sum += self.pi[reviewNum][lineNum][aspect] * self.topic_model[aspect][word]
                    for aspect in self.topic_model:
                        self.hidden_parameters[reviewNum][lineNum][word][aspect] = self.pi[reviewNum][lineNum][aspect] * \
                                                                                   self.topic_model[aspect][
                                                                                       word] / my_sum
                        self.hidden_parameters_background[reviewNum][lineNum][word] = (self.lambda_background *
                                                                                       self.background_probability[
                                                                                           word]) / (
                                                                                                  self.lambda_background *
                                                                                                  self.background_probability[
                                                                                                      word] + ((
                                                                                                                           1 - self.lambda_background) * my_sum))

    def m_step(self):
        print(type(self).__name__, '- m_step...')

    def _dump_hidden_parameters(self):
        print(type(self).__name__, '- _dump_hidden_parameters...')
        np.save(self.dump_path + "MY_HP_Updated", self.hidden_parameters)
        np.save(self.dump_path + "MY_HPB_updated", self.hidden_parameters_background)


if __name__ == '__main__':
    em = ExpectationMinimizationOriginal()
    em.em()
