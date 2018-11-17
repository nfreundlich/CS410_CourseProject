import time

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
        self.v = 0
        self.f = 0

        # Parameters computed from collection
        self.reviews = None
        self.topic_model = None
        self.background_probability = None

        # Parameters for EM computation
        self.hidden_parameters = None
        self.hidden_parameters_background = None
        self.pi = None

    def import_data(self):
        print(type(self).__name__, "- base import data...")

    def initialize_parameters(self):
        print(type(self).__name__, "- base initialize parameters...")

    def em_loop(self):
        print(type(self).__name__, "- base loop...")
        while self.max_iter:
            start_time_iteration = time.time()
            self.max_iter -= 1
            self.e_step()
            self.m_step()
            end_time_iteration = time.time()
            print("Elapsed on iteration: {} seconds".format(round(end_time_iteration - start_time_iteration, 4)))
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
        start_time = time.time()

        self.import_data()
        self.initialize_parameters()
        self.em_loop()

        end_time = time.time()
        print("Elapsed: {} seconds".format(round(end_time - start_time, 4)))

    def _dump_hidden_parameters(self):
        print(type(self).__name__, "- base dump hidden parameters...")


if __name__ == '__main__':
    em = ExpectationMaximization()
    em.em()
