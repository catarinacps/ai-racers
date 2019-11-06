import numpy as np
from datetime import datetime
from controller import controller
import pickle
import os.path


class Controller(controller.Controller):
    def __init__(self, track, evaluate=True, bot_type=None):
        super().__init__(track, evaluate=evaluate, bot_type=bot_type)

        self.algorithm = "Hill Climbing"
        self.alias = "hc"

    def local_search(self, initial_weights, *argv):

        return self.hill_climbing_new(initial_weights, *argv)

    def save_result(self, weights, score):

        f = open(self.alias + "-" + "best_w", "w+")
        f.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S") + "\r\n")
        f.write(super().featureNames)
        f.write("Score: ", score)
        f.write(weights)
        f.close()





    def hill_climbing_new(self, weights, percentage=0.5):

        percentage = float(percentage)

        if (os.path.exists('hc_previous_best.pkl')):
            with open('hc_previous_best.pkl', 'rb') as parameters_file, open('hc_previous_info.pkl', 'rb') as info_file:
                best_parameters = pickle.load(parameters_file)
                best_score, iteration, improvement, desespero = pickle.load(info_file)

        else:
            best_parameters = weights
            iteration = 1
            best_score = self.run_episode(weights)
            improvement = best_score
            desespero = 1

        print("\n[Iter ", iteration, "]")
        cur_score = best_score
        cur_parameters = list(best_parameters)
        changed = False

        # exhaustively generate neighbors based on input percentage
        for i, w in enumerate(cur_parameters):
            for sign in [1, -1]:

                # neighbor is a small (positive or negative) perturbation in one weight
                neighbor = list(best_parameters)
                neighbor[i] += sign*percentage*desespero
                new_score = self.run_episode(neighbor)

                if new_score > best_score:
                    changed = True
                    desespero = 1
                    best_score = new_score
                    best_parameters = neighbor
                    print("\n[Iter ", iteration, " Weight ", i, " Sign ",sign, "] New best: ", best_score)

        iteration += 1
        improvement = best_score - cur_score
        if not changed:
            desespero += 0.5

        with open('hc_previous_best.pkl', 'wb') as parameters_file:
            pickle.dump(best_parameters, parameters_file)

        with open('hc_previous_info.pkl', 'wb') as info_file:
            pickle.dump([best_score, iteration, improvement, desespero], info_file)

        return best_parameters, best_score




    # OLD CMA-ES WITH COUPLED LOOP

    # Input initial weights, percentage perturbance
    # Output better weights
    def hill_climbing(self, weights, percentage=0.5):

        percentage = float(percentage)

        best_score = self.run_episode(weights)
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