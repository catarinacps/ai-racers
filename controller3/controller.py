import numpy as np
from datetime import datetime
from controller import controller


class Controller(controller.Controller):
    def __init__(self, track, evaluate=True, bot_type=None):
        super().__init__(track, evaluate=evaluate, bot_type=bot_type)

        self.algorithm = "Hill Climbing"
        self.alias = "hc"

    def local_search(self, initial_weights):

        self.hill_climbing(initial_weights)

    def save_result(self, weights, score):

        f = open(self.alias + "-" + "best_w", "w+")
        f.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S") + "\r\n")
        f.write(super().featureNames)
        f.write("Score: ", score)
        f.write(weights)
        f.close()

    # Input initial weights, percentage perturbance
    # Output better weights

    def hill_climbing(self, weights, num_neighbors = 8, dummy_param = 0, percentage = 0.5):

        # TODO: change the way we get neighbors!

        best_score = self.run_episode(weights)
        best_parameters = weights

        # exhaustively generate neighbors based on input percentage
        for i, weight in enumerate(weights):
            for sign in [1, -1]:
                # neighbor is a small (positive or negative) perturbation in one weight
                neighbor = list(weights)
                neighbor[i] += sign * (weight*percentage)

                new_score = self.run_episode(neighbor)
                if new_score > best_score:
                    best_score = new_score
                    best_parameters = neighbor

        return np.array(best_parameters), best_score

