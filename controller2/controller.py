import controller_template as controller_template
import numpy as np
from math import ceil

# Pretend these are constant indexes for sensors[]
DIST_LEFT = 0
DIST_CENTER = 1
DIST_RIGHT = 2
ON_TRACK = 3
DIST_CHECKPOINT = 4
SPEED = 5
DIST_ENEMY = 6
ENEMY_ANGLE = 7
ENEMY_NEAR = 8
NUM_OF_ACTIONS = 5
BIG_NUMBER = 1000000

class Controller(controller_template.Controller):
    def __init__(self, track, evaluate=True, bot_type=None):
        super().__init__(track, evaluate=evaluate, bot_type=bot_type)

        self.num_actions = 5
        self.sensors = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.prev_st = [0, 0, 0, 0, 0, 0.1, 0, 0, 0]
        self.prev_score = -1000


        self.features = [1, 0, 0, 0, 0, 0]
        self.num_features = len(self.features)



    #######################################################################
    ##### METHODS YOU NEED TO IMPLEMENT ###################################
    #######################################################################

    def take_action(self, parameters: list) -> int:
        """
        :param parameters: Current weights/parameters of your controller
        :return: An integer corresponding to an action:
        1 - Right
        2 - Left
        3 - Accelerate
        4 - Brake
        5 - Nothing
        """

        features = self.compute_features(self.sensors)
        actions = []
        par_each_Q = len(parameters) // NUM_OF_ACTIONS

        weights = np.reshape(np.array(parameters), (NUM_OF_ACTIONS, par_each_Q))


        for param in weights:
            actions.append(
                np.sum(np.array(features) * np.array(param))
            )


        best_action = np.argmax(np.array(actions)) + 1

        return best_action


    def compute_features(self, st):
        """
        :param sensors: Car sensors at the current state s_t of the race/game
        contains (in order):
            track_distance_left: 1-100
            track_distance_center: 1-100
            track_distance_right: 1-100
            on_track: 0 or 1
            checkpoint_distance: 0-???
            car_velocity: 10-200
            enemy_distance: -1 or 0-???
            position_angle: -180 to 180
            enemy_detected: 0 or 1
          (see the specification file/manual for more details)
        :return: A list containing the features you defined
        """


        diffCheckpoint = st[DIST_CHECKPOINT] - self.prev_st[DIST_CHECKPOINT]
        diffCheckpoint = diffCheckpoint*2 / 1000
        # times 2 cause it is double important

        # goal to minimize getting on grass.
        # if not on track, adds spoiler constant.
        riskHeadCollision = (1 - st[ON_TRACK])*250 + (st[SPEED] / st[DIST_CENTER])*1.5
        # 250 and not 200 to say that head collision is worse than right/left
        # the same logic goes to the "1.5"

        # without considering the constants multiplication:
        # max value: 200 + 10/1 = 210
        # min value: 0 + 10/100 = 1/10 = 0.1
        riskHeadCollision = (riskHeadCollision - 0.1) / 209.9   #normalized

        riskLeftCollision = (1 - st[ON_TRACK])*200 + (st[SPEED] / st[DIST_LEFT])
        riskLeftCollision = (riskLeftCollision - 0.1) / 209.9

        riskRightCollision = (1 - st[ON_TRACK])*200 + (st[SPEED] / st[DIST_RIGHT])
        riskRightCollision = (riskRightCollision - 0.1) / 209.9


        centralizedPosition = (abs(st[DIST_LEFT] - st[DIST_RIGHT]) * -1) + 99
        # max: 0 * -1 + 99 = 99
        # min: |100 - 1| * -1 + 99 = 0
        centralizedPosition = centralizedPosition / 99



        #if (self.game_state.car1.score > self.prev_score)
        # depois vejo como lidar quando cruza checkpoint
        # maximize potential travel distance fulfilled
        # speed is distance units per 10 frames
        # if distance to checkpoint reduced by a similar amount,
        # then the car is going straight to target and ratio == 1.0
        # opposite direction: ratio -1.0
        #maxTravel = st[SPEED]/10
        #direction_kinda = -diffCheckpoint/maxTravel

        self.prev_st = st
        return [1, diffCheckpoint, riskHeadCollision, riskLeftCollision,
        riskRightCollision, centralizedPosition]


    def learn(self, weights, args = None) -> list:
        """
        IMPLEMENT YOUR LEARNING METHOD (i.e. YOUR LOCAL SEARCH ALGORITHM) HERE

        HINT: you can call self.run_episode (see controller_template.py) to evaluate a given set of weights
        :param weights: initial weights of the controller (either loaded from a file or generated randomly)
        :return: the best weights found by your learning algorithm, after the learning process is over
        """


        print("\n\n############### STARTING TRAINING ###############\n\n")
        if args is not None:
            best_weights = self.cma_es(weights, args[0], args[1])
        else:
            best_weights = self.cma_es(weights)
        print("\n\n############### BEST WEIGHTS ###############n\n")
        print(best_weights)
        np.savetxt("cmaes_best_w.txt", np.array(best_weights))
        return

        #raise NotImplementedError("This Method Must Be Implemented")


    # Covariance Matrix Adaptation Evolution Strategy
    # Input initial weights
    # Output better weights
    def cma_es(self, weights, sample_size = 5, top_percentage = 0.5):

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

        return list(best_parameters)


    # vou deixar essa Coisa aqui ate achar um jeito pratico de usar
    @property
    def distLeft(self):
        return self.sensors[0]

    @property
    def distCenter(self):
        return self.sensors[1]

    @property
    def distRight(self):
        return self.sensors[2]

    @property
    def onTrack(self):
        return self.sensors[3]
    @property
    def distCheckpoint(self):
        return self.sensors[4]

    @property
    def speed(self):
        return self.sensors[5]

    @property
    def distEnemy(self):
        return self.sensors[6]

    @property
    def enemyAngle(self):
        return self.sensors[7]

    @property
    def enemyNearby(self):
        return self.sensors[8]

    @distLeft.setter
    def distLeft(self, val):
        self.sensors[0] = val

    @distCenter.setter
    def distCenter(self, val):
        self.sensors[1] = val

    @distRight.setter
    def distRight(self, val):
        self.sensors[2] = val

    @onTrack.setter
    def onTrack(self, val):
        self.sensors[3] = val

    @distCheckpoint.setter
    def distCheckpoint(self, val):
        self.sensors[4] = val

    @speed.setter
    def speed(self, val):
        self.sensors[5] = val

    @distEnemy.setter
    def distEnemy(self, val):
        self.sensors[6] = val

    @enemyAngle.setter
    def enemyAngle(self, val):
        self.sensors[7] = val

    @enemyNearby.setter
    def enemyNearby(self, val):
        self.sensors[8] = val
