import numpy as np
import math
from feature_mining.em_base import ExpectationMaximization


class ExpectationMaximizationOriginal(ExpectationMaximization):
    """
    Original EM Algorithm as developed by Santu.
    """

    def __init__(self, dump_path="../tests/data/em_01/"):
        print(type(self).__name__, '- init...')
        ExpectationMaximization.__init__(self, dump_path=dump_path)
        self.previous_pi = []

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
                        self.hidden_parameters_background[reviewNum][lineNum][word] = \
                            (self.lambda_background * self.background_probability[word]) / \
                            (self.lambda_background * self.background_probability[word] + ((1 - self.lambda_background) * my_sum))

    def m_step(self):
        print(type(self).__name__, '- m_step...')

        self.previous_pi = []
        for reviewNum in range(0, len(self.reviews)):
            self.previous_pi.append(list())
            for lineNum in range(0, len(self.reviews[reviewNum])):
                self.previous_pi[reviewNum].append({})
                for aspect in self.topic_model:
                    self.previous_pi[reviewNum][lineNum][aspect] = self.pi[reviewNum][lineNum][aspect]

        for reviewNum in range(0, len(self.reviews)):
            for lineNum in range(0, len(self.reviews[reviewNum])):
                denom = 0
                for aspect in self.topic_model:
                    for word in self.reviews[reviewNum][lineNum]:
                        denom += self.reviews[reviewNum][lineNum][word] * (1 - self.hidden_parameters_background[reviewNum][lineNum][word]) * \
                                 self.hidden_parameters[reviewNum][lineNum][word][aspect]
                # np.save(self.dump_path + "DENOM", denom)

                for aspect in self.topic_model:
                    nom = 0
                    for word in self.reviews[reviewNum][lineNum]:
                        nom += self.reviews[reviewNum][lineNum][word] * (1 - self.hidden_parameters_background[reviewNum][lineNum][word]) * \
                               self.hidden_parameters[reviewNum][lineNum][word][aspect]

                    # np.save(self.dump_path + "NOM", nom)
                    try:
                        self.pi[reviewNum][lineNum][aspect] = nom / denom
                    except:
                        print(reviewNum, lineNum, aspect, nom, denom)


    def compute_cost(self):
        #self.pi = np.load(self.dump_path + "PI_updated.npy")

        dist = 0.0
        for reviewNum in range(0, len(self.reviews)):
            for lineNum in range(0, len(self.reviews[reviewNum])):
                for aspect in self.topic_model:
                    dist = dist + math.pow(self.pi[reviewNum][lineNum][aspect] - self.previous_pi[reviewNum][lineNum][aspect], 2)

        print('dist=' + str(dist))
        np.save(self.dump_path + "MY_DIST", dist)
        return 0.0

    def _dump_hidden_parameters(self):
        print(type(self).__name__, '- _dump_hidden_parameters...')
        np.save(self.dump_path + "MY_HP_Updated", self.hidden_parameters)
        np.save(self.dump_path + "MY_HPB_updated", self.hidden_parameters_background)
        np.save(self.dump_path + "MY_PI_updated", self.pi)
        np.save(self.dump_path + "MY_PREVIOUS_PI", self.previous_pi)


if __name__ == '__main__':
    em = ExpectationMaximizationOriginal()
    em.em()
