import numpy as np
from datetime import datetime
from controller import controller


class Controller(controller.Controller):
    def __init__(self, track, evaluate=True, bot_type=None):
        super().__init__(track, evaluate=evaluate, bot_type=bot_type)

        self.algorithm = "Hill Climbing"
        self.alias = "hc"

    def local_search(self, initial_weights):

        return self.hill_climbing(initial_weights)

    def save_result(self, weights, score):

        f = open(self.alias + "-" + "best_w", "w+")
        f.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S") + "\r\n")
        f.write(super().featureNames)
        f.write("Score: ", score)
        f.write(weights)
        f.close()

    # Input initial weights, percentage perturbance
    # Output better weights

    def hill_climbing(self, weights, num_neighbors = 8, dummy_param = 0, percentage=0.1):

        # TODO: change the way we get neighbors!

        best_score = self.run_episode(weights)
        #best_parameters = [1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4,5,6]
        best_parameters = weights
        max_iterations = 100
        iteration = 1
        improvement = best_score
        desespero = 1
        while improvement > 0 or iteration <= max_iterations:
            print("\n[Iter ", iteration, "]")
            cur_score = best_score
            cur_parameters = list(best_parameters)
            changed = False
            print(cur_parameters)
            print("Scores found: "),
            # exhaustively generate neighbors based on input percentage
            for i, w in enumerate(cur_parameters):
                for sign in [1, -1]:
                    # neighbor is a small (positive or negative) perturbation in one weight
                    neighbor = list(best_parameters)
                    neighbor[i] += sign*0.5*desespero
                    new_score = self.run_episode(neighbor)
                    print(new_score, end=" ")

                    if new_score > best_score:
                        changed = True
                        desespero = 1
                        best_score = new_score
                        best_parameters = neighbor
                        print("\n[Iter ", iteration, " Weight ", i, " Sign ",sign, "] New best: ", best_score)
                        print(best_parameters, "\n")

            iteration += 1
            improvement = best_score - cur_score
            if not changed:
                desespero += 0.5

        return best_parameters, best_score

