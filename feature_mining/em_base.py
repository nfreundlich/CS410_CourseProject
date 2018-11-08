class ExpectationMaximization:
    """
    Base class for Expectation - Maximization algorithm.
    Treats data preparation, looping, e-step and m-step.
    """

    def __init__(self, dump_path="../tests/data/em_01/"):
        print(type(self).__name__, "- base init...")

        # Path used for testing
        self.dump_path = dump_path

        # Parameters for EM execution control
        self.max_iter = 50
        self.lambda_background = 0.7
        self.dist_threshold = 1e-6

        # Parameters related to collection size
        self.m = 0
        self.nw = 0
        self.na = 0

        # Parameters computed from collection
        self.reviews = None  # np.load(self.dump_path + "Reviews.npy")
        self.topic_model = None  # np.load(self.dump_path + 'TopicModel.npy').item()
        self.background_probability = None  # np.load(self.dump_path + 'BackgroundProbability.npy').item()

        # Parameters for EM computation
        self.hidden_parameters = None  # np.load(self.dump_path + "HP.npy")
        self.hidden_parameters_background = None  # np.load(self.dump_path + "HPB.npy")
        self.pi = None  # np.load(self.dump_path + "PI.npy")

        # Parameters for matrix result interpretation
        self.aspects_map = {}
        self.words_map = {}

    def import_data(self):
        print(type(self).__name__, "- base import data...")

    def initialize_parameters(self):
        print(type(self).__name__, "- base initialize parameters...")

    def em_loop(self):
        print(type(self).__name__, "- base loop...")
        while self.max_iter:
            self.max_iter -= 1
            self.e_step()
            self.m_step()
            if self.compute_cost() < self.dist_threshold:
                break

    def e_step(self):
        print(type(self).__name__, "- base e_step...")

    def m_step(self):
        print(type(self).__name__, "- base m_step...")

    def compute_cost(self):
        print(type(self).__name__, "- base compute cost...")
        return 0.0

    def em(self):
        self.import_data()
        self.initialize_parameters()
        self.em_loop()

    def _dump_hidden_parameters(self):
        print(type(self).__name__, "- base dump hidden parameters...")


if __name__ == '__main__':
    em = ExpectationMaximization()
    em.em()
