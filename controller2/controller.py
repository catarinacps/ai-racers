import numpy as np
import os.path
import pickle
from math import ceil
from datetime import datetime
from controller import controller


class Controller(controller.Controller):
    def __init__(self, track, evaluate=True, bot_type=None):
        super().__init__(track, evaluate=evaluate, bot_type=bot_type)

        self.algorithm = "CMA-ES"
        self.alias = "cma"

    def local_search(self, initial_weights, *argv):
        return self.cma_es_new(initial_weights, *argv)

    def save_result(self, weights, score):
        f = open(self.alias + "-" + "best_w", "w+")
        f.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S") + "\r\n")
        f.write(super().featureNames)
        f.write("Score: ", score)
        f.write(weights)
        f.close()


    def cma_es_new(self,
                   weights,
                   sample_size=25,
                   sigma=0.5,
                   top_percentage=0.5,
                   convergence_delta=0.01):

        sample_size = int(sample_size)
        sigma = float(sigma)
        top_percentage = float(top_percentage)
        convergence_delta = float(convergence_delta)

        # Algorithm parameters
        # sample_size: number of randomly generated candidates
        # weights: current mean around which to generate neighbors
        best_of = ceil(sample_size * top_percentage)
        n_weights = len(weights)

        if os.path.exists('cmaes_previous_mean.pkl'):
            with open('cmaes_previous_mean.pkl', 'rb') as prev_file:
                mean = pickle.load(prev_file)
            with open('cmaes_previous_best.pkl', 'rb') as prev_file:
                iteration, best_score, best_parameters = pickle.load(prev_file)
        else:
            mean = np.asarray(weights)
            best_score = self.run_episode(weights)
            best_parameters = weights
            iteration = 1

        # Covariance matrix
        # média é o best_score dos pesos iniciais? errou dá zero pra ela
        # initial mean: N-dimensional array of current parameters

        # initial sigma step > 0
        # initial covariance matrix:
        # NxN-dimensional symmetric matrix with all-positive eigs
        covariance = np.identity(n_weights)
        p0 = np.zeros(n_weights)
        p1 = np.zeros(n_weights)
        print("Initial mean weights: ")
        print(mean)
        print("Initial sigma: ", sigma)
        print("Initial covariance matrix:")
        print(sigma ** 2 * covariance)


        # Neighbor generation
        neighbors = np.empty([sample_size, n_weights])
        neighbor_scores = np.empty(sample_size)
        improvement = 0

        for i in range(sample_size):

            neighbors[i] = np.random.multivariate_normal(mean, (sigma ** 2) * covariance)
            neighbor_scores[i] = self.run_episode(list(neighbors[i]))


        # sort neighbors by descending score
        order = neighbor_scores.argsort()[::-1]
        neighbor_scores = neighbor_scores[order]
        neighbors = neighbors[order]

        # top neighbors
        best_neighbors = neighbors[0:best_of]
        best_neighbor_scores = neighbor_scores[0:best_of]

        # new covariance
        cc = np.zeros([n_weights, n_weights])
        for neighbor in best_neighbors:

            a = neighbor - mean
            ccc = a.reshape(n_weights, 1) @ a.reshape(1, n_weights)
            cc = cc + ccc


        cc = np.divide(cc, np.size(best_neighbors))


        new_mean = np.mean(best_neighbors, axis=0, dtype=np.float64)
        mean = new_mean

        # atualiza a média com todos os vizinhos existentes e os novos - ok
        # tira os bad e reshape a covariancia? - ok maybe
        # atualiza parametros de sampling para proxima iteracao - nao sei
        # improving = ta melhorando a media? entao roda de novo que ta ficando bom - ok
        improvement = best_neighbor_scores[0] - best_score

        if best_score < best_neighbor_scores[0]:
            best_score = best_neighbor_scores[0]
            best_parameters = list(best_neighbors[0])
            print("New best score: ", best_score)
            print("New best candidate")
            print(best_parameters)
            print("Improvement: ", improvement)

        iteration += 1

        with open('cmaes_previous_mean.pkl', 'wb') as new_mean_file:
            pickle.dump(mean, new_mean_file)
        with open('cmaes_previous_best.pkl', 'wb') as new_best_file:
            pickle.dump([iteration, best_score, best_parameters], new_best_file)

        return list(best_parameters), best_score



    # OLD CMA-ES WITH COUPLED LOOP

    # Covariance Matrix Adaptation Evolution Strategy
    # Input initial weights
    # Output better weights
    def cma_es(self,
               weights,
               sample_size=5,
               top_percentage=0.5,
               convergence_delta=0.01):

        sample_size = int(sample_size)
        top_percentage = float(top_percentage)
        convergence_delta = float(convergence_delta)

        # Algorithm parameters
        # sample_size: number of randomly generated candidates
        # weights: current mean around which to generate neighbors
        best_of = ceil(sample_size * top_percentage)
        convergence_delta = 0.01
        max_iter = 50
        n_weights = len(weights)

        # Covariance matrix
        # média é o best_score dos pesos iniciais? errou dá zero pra ela
        # initial mean: N-dimensional array of current parameters
        mean = np.asarray(weights)
        # initial sigma step > 0
        sigma = 0.5
        # initial covariance matrix:
        # NxN-dimensional symmetric matrix with all-positive eigs
        covariance = np.identity(n_weights)
        p0 = np.zeros(n_weights)
        p1 = np.zeros(n_weights)
        print("Initial mean weights: ")
        print(mean)
        print("Initial sigma: ", sigma)
        print("Initial covariance matrix:")
        print(sigma ** 2 * covariance)


        # Neighbor generation
        neighbors = np.empty([sample_size, n_weights])
        neighbor_scores = np.empty(sample_size)
        best_score = self.run_episode(weights)
        best_parameters = weights
        iter = 0
        improvement = 1
        while (iter < max_iter) or (improvement > convergence_delta):
            iter += 1
            print("Iteration ", iter, ":")

            for i in range(sample_size):

                neighbors[i] = np.random.multivariate_normal(mean, (sigma ** 2) * covariance)
                neighbor_scores[i] = self.run_episode(list(neighbors[i]))


            order = neighbor_scores.argsort()[::-1]
            neighbor_scores = neighbor_scores[order]
            neighbors = neighbors[order]

            # top neighbors
            best_neighbors = neighbors[0:best_of]
            best_neighbor_scores = neighbor_scores[0:best_of]

            # new covariance
            cc = np.zeros([n_weights, n_weights])
            for neighbor in best_neighbors:
                a = neighbor - mean
                ccc = a.reshape(n_weights, 1) @ a.reshape(1, n_weights)
                cc = cc + ccc


            cc = np.divide(cc, np.size(best_neighbors))


            new_mean = np.mean(best_neighbors, axis=0, dtype=np.float64)
            mean = new_mean


            improvement = best_neighbor_scores[0] - best_score

            if best_score < best_neighbor_scores[0]:
                best_score = best_neighbor_scores[0]
                best_parameters = list(best_neighbors[0])
                print("New best score: ", best_score)
                print("New best candidate")
                print(best_parameters)
                print("Improvement: ", improvement)

        return list(best_parameters), best_score