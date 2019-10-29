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

class Controller(controller_template.Controller):
    def __init__(self, track, evaluate=True, bot_type=None):
        super().__init__(track, evaluate=evaluate, bot_type=bot_type)

        self.num_actions = 5
        self.sensors = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.prev_st = [0, 0, 0, 0, 0, 0.1, 0, 0, 0]
        self.prev_score = -1000

        # diffCheckpoint, riskHeadCollision, direction_kinda
        self.features = [ 0, 0, 0 ]
        self.num_features = len(self.features)

        weights = np.zeros( (self.num_actions, self.num_features) )


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
        print("Computed features: ", features)

        preference = -1
        highest = -1
        #for i in range(self.num_features):

        #     action[i] = np.dot(weights[i],features) # falta o peso constante
        #     if action[i] > highest:
        #         highest = action[i]
        #         preference = i
        #
        #
        # return i + 1

        # when in doubt respect the speed limit :)
        return 5

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

        print("Previous state: ", self.prev_st)
        print("Current state: ", st)

        diffCheckpoint = st[DIST_CHECKPOINT] - self.prev_st[DIST_CHECKPOINT]
        # goal to minimize getting on grass.
        # if not on track, adds spoiler constant.
        riskHeadCollision = (1 - st[ON_TRACK])*1000 + (st[SPEED] / st[DIST_CENTER] )

        #if (self.game_state.car1.score > self.prev_score)
        # depois vejo como lidar quando cruza checkpoint
        # maximize potential travel distance fulfilled
        # speed is distance units per 10 frames
        # if distance to checkpoint reduced by a similar amount,
        # then the car is going straight to target and ratio == 1.0
        # opposite direction: ratio -1.0
        maxTravel = st[SPEED]/10
        direction_kinda = -diffCheckpoint/maxTravel

        self.prev_st = st
        return [diffCheckpoint, riskHeadCollision, direction_kinda]


    def learn(self, weights) -> list:
        """
        IMPLEMENT YOUR LEARNING METHOD (i.e. YOUR LOCAL SEARCH ALGORITHM) HERE

        HINT: you can call self.run_episode (see controller_template.py) to evaluate a given set of weights
        :param weights: initial weights of the controller (either loaded from a file or generated randomly)
        :return: the best weights found by your learning algorithm, after the learning process is over
        """
        raise NotImplementedError("This Method Must Be Implemented")

    # Input initial weights, percentage perturbance
    # Output better weights
    def hill_climbing(self, weights, percentage):

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

        return best_parameters


    # Covariance Matrix Adaptation Evolution Strategy
    # Input initial weights
    # Output better weights
    def cma_es(self, weights, sample_size=5):

        # Algorithm parameters
        # sample_size: number of randomly generated candidates
        # weights: current mean around which to generate neighbors
        best_of = ceil(sample_size / 2)
        stop_converged = 0.01
        stop_iter = 10000
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
        print("Initial paths:")
        print(p0)
        print(p1)

        # Neighbor generation
        neighbors = np.empty([sample_size, n_weights])
        neighbor_scores = np.empty(sample_size)
        best_score = self.run_episode(weights)
        best_parameters = weights
        t = True
        while t:
            for i in range(sample_size):
                print("Iter ", i)
                neighbors[i] = np.random.multivariate_normal(mean, (sigma ** 2) * covariance)
                print("Sampled: ", neighbors[i])
                neighbor_scores[i] = self.run_episode(list(neighbors[i]))
                # np.random.randint(-50, 100)
                print("Score: ", neighbor_scores[i])

            print("Finished sampling neighbors")
            print("CANDIDATES: ", neighbors)
            print("SCORES: ", neighbor_scores)
            # sort neighbors by descending score
            order = neighbor_scores.argsort()[::-1]
            neighbor_scores = neighbor_scores[order]
            neighbors = neighbors[order]
            print("ARGSORT: ", order)
            print("SORTED CANDIDATES: ", neighbors)
            print("SORTED SCORES: ", neighbor_scores)
            t = False

            # top neighbors
            neighbors = neighbors[0:best_of]
            neighbor_scores = neighbor_scores[0:best_of]
            new_mean = np.mean(neighbors, axis=0, dtype=np.float64)
            print("New mean (first ", best_of, " neighbors): ")
            print(new_mean)

            # new covariance

            cc = np.zeros([n_weights, n_weights])
            for neighbor in neighbors:
                print("Neighbor da vez")
                print(neighbor)
                print("Neighbor menos mean")
                print(neighbor - mean)
                a = neighbor - mean
                ccc = a.reshape(n_weights, 1) @ a.reshape(1, n_weights)
                print("Multiply")
                print(ccc)
                cc = cc + ccc

            print("New covariance???")
            cc = np.divide(cc, np.size(neighbors))
            print(cc)

            # atualiza a média com todos os vizinhos existentes e os novos - ok
            # tira os bad e reshape a covariancia? - ok maybe
            # atualiza parametros de sampling para proxima iteracao - nop
            # improving = ta melhorando a media? entao roda de novo que ta ficando bom - nop

        return  # argsort dos melhores candidatos e pega o parametro do melhor


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