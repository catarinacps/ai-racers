import numpy as np
from math import ceil
from datetime import datetime
from controller import controller


class Controller(controller.Controller):
    def __init__(self, track, evaluate=True, bot_type=None):
        super().__init__(track, evaluate=evaluate, bot_type=bot_type)

        self.algorithm = "CMA-ES"
        self.alias = "cma"

    def local_search(self, initial_weights):
        return self.cma_es(initial_weights)

    def save_result(self, weights, score):
        # will probably be inherited away
        f = open(self.alias + "-" + "best_w", "w+")
        f.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S") + "\r\n")
        f.write(super().featureNames)
        f.write("Score: ", score)
        f.write(weights)
        f.close()

    # Covariance Matrix Adaptation Evolution Strategy
    # Input initial weights
    # Output better weights
    def cma_es(self,
               weights,
               sample_size=5,
               top_percentage=0.5,
               convergence_delta=0.01):

        # Algorithm parameters
        # sample_size: number of randomly generated candidates
        # weights: current mean around which to generate neighbors
        best_of = ceil(sample_size * top_percentage)
        convergence_delta = 0.01
        min_iter = 10
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
        # print("Initial paths:")
        # print(p0)
        # print(p1)

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
                # print("Sample ", i)
                neighbors[i] = np.random.multivariate_normal(mean, (sigma ** 2) * covariance)
                # print("Sampled: ", neighbors[i])
                neighbor_scores[i] = self.run_episode(list(neighbors[i]))
                # np.random.randint(-50, 100)
                # print("Score: ", neighbor_scores[i])

            # print("Finished sampling neighbors")
            # print("CANDIDATES: ", neighbors)
            # print("SCORES: ", neighbor_scores)
            # sort neighbors by descending score
            order = neighbor_scores.argsort()[::-1]
            neighbor_scores = neighbor_scores[order]
            neighbors = neighbors[order]
            # print("ARGSORT: ", order)
            # print("SORTED CANDIDATES: ", neighbors)
            # print("SORTED SCORES: ", neighbor_scores)
            # top neighbors
            best_neighbors = neighbors[0:best_of]
            best_neighbor_scores = neighbor_scores[0:best_of]

            # new covariance

            cc = np.zeros([n_weights, n_weights])
            for neighbor in best_neighbors:
                # print("Neighbor da vez")
                # print(neighbor)
                # print("Neighbor menos mean")
                # print(neighbor - mean)
                a = neighbor - mean
                ccc = a.reshape(n_weights, 1) @ a.reshape(1, n_weights)
                # print("Multiply")
                # print(ccc)
                cc = cc + ccc

            # print("New covariance???")
            cc = np.divide(cc, np.size(best_neighbors))
            # print(cc)

            new_mean = np.mean(best_neighbors, axis=0, dtype=np.float64)
            # print("New mean (first ", best_of, " neighbors): ")
            # print(new_mean)
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

        return list(best_parameters), best_score
