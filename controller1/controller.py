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

class Controller(controller_template.Controller):
    def __init__(self, track, evaluate=True, bot_type=None):
        super().__init__(track, evaluate=evaluate, bot_type=bot_type)

        self.num_actions = 5
        self.sensors = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.prev_st = [0, 0, 0, 0, 0, 0.1, 0, 0, 0]
        self.prev_score = -1000

        
        self.features = [1, 0, 0, 0, 0, 0, 0]
        self.num_features = len(self.features)

        # experimental to normalize diff feature
        self.smallest_chkp_diff = 10000000

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
        actions = []
        par_each_Q = len(parameters) // NUM_OF_ACTIONS

        weights = np.reshape(np.array(parameters), (NUM_OF_ACTIONS, par_each_Q))
        
        '''
        if (par_each_Q > self.num_features): # means that the first parameter must be summed up
                                             # without multiplying
            for param in weights:
                actions.append(
                    param[0] + np.sum(np.array(features) * np.array(param[1:]))
                )
        
        else:
        '''

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
        diffCheckpoint = diffCheckpoint / 1000

        # experimental
        if diffCheckpoint < self.smallest_chkp_diff:
            self.smallest_chkp_diff = diffCheckpoint


        # goal to minimize getting on grass.
        # if not on track, adds spoiler constant.
        riskHeadCollision = (1 - st[ON_TRACK])*200 + (st[SPEED] / st[DIST_CENTER])
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
        maxTravel = st[SPEED]/10
        direction_kinda = -diffCheckpoint/maxTravel

        self.prev_st = st
        return [1, diffCheckpoint, riskHeadCollision, riskLeftCollision, 
        riskRightCollision, direction_kinda, centralizedPosition]


    def learn(self, weights) -> list:
        """
        IMPLEMENT YOUR LEARNING METHOD (i.e. YOUR LOCAL SEARCH ALGORITHM) HERE

        HINT: you can call self.run_episode (see controller_template.py) to evaluate a given set of weights
        :param weights: initial weights of the controller (either loaded from a file or generated randomly)
        :return: the best weights found by your learning algorithm, after the learning process is over
        """


        print("\n\n############### STARTING TRAINING ###############\n\n")
        best_weights = self.genetic_algorithm(weights)
        print("\n\n############### BEST WEIGHTS ###############n\n")
        print(best_weights)
        np.savetxt("ga_best_w.txt", np.array(best_weights))
        return

        #raise NotImplementedError("This Method Must Be Implemented")

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

        return np.array(best_parameters)


    # Covariance Matrix Adaptation Evolution Strategy
    # Input initial weights
    # Output better weights
    def cma_es(self, weights, sample_size=5):

        # Algorithm parameters
        # sample_size: number of randomly generated candidates
        # weights: current mean around which to generate neighbors
        best_of = ceil(sample_size / 2)
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

    def genetic_algorithm(self, weights):

            population_size = 100
            elitism = 0.1
            roulette = 0.1
            mutation_rate = 0.2
            max_generations = 500
            max_same_best = 10
            perturbation_range = 2 # because weights got big (empirical experimentation)

            population = self.generate_population(weights, population_size)
            fitness = self.compute_fitness(population)

            generation = 1
            same_best = 0

            best_idx = np.argmax(fitness)
            best_individual_prev = population[best_idx]

            while generation <= max_generations:
                
                print("\n\nGeneration:", generation)
                print("Best score:", fitness[best_idx])
                print("Smallest checkpoint diff:", self.smallest_chkp_diff)
                print("\n\n")
                np.savetxt("best_genetic.txt", best_individual_prev)

                population = self.select_population(population, fitness, elitism, roulette)
                population = self.cross_population(population, population_size)
                population = self.mutate_population(population, mutation_rate)
                fitness = self.compute_fitness(population)
                generation += 1

                best_idx = np.argmax(fitness)
                best_individual = population[best_idx]

                if (np.array_equal(best_individual, best_individual_prev)):
                    same_best += 1
                    if (same_best >= max_same_best):
                        print("Same individual found", same_best,
                            "times. Learning algorithm stopped.")
                        return population[np.argmax(fitness)]

                else:
                    same_best = 0

                best_individual_prev = best_individual

            print("Max generations reached. Learning algorithm stopped.")
            return population[np.argmax(fitness)]


    # Input parameters: list size of the current weights; number of individuals to be generated
    # Output returned: new set of weights
    def generate_population(self, weights, population_size):
        
        population = [weights]

        for i in range(population_size-1):
            population.append(
                #np.random.uniform(low=np.amin(weights), high=np.amax(weights), size=len(weights))
                np.random.uniform(low=-1.0, high=1.0, size=len(weights))
            )

        return population


    # Input parameters: list of individual solutions
    # Output returned: list of fitness score for each individual
    def compute_fitness(self, population):
        BIG_NUMBER = 1000000
        fitness = []

        for individual in population:
            fitness.append(self.run_episode(individual) + BIG_NUMBER)

        return fitness


    # Input parameters: list of individual solutions; fitness; fraction to keep by elitism; fraction to keep by roulette
    # Output returned: selected individuals
    def select_population(self, population, fitness, elitism, roulette):

        
        fitness = np.array(fitness)
        population = np.array(population)

        elite = []
        if (elitism != 0):

            num_from_elite = round(elitism * len(population))
            elite_idx = np.argpartition(fitness, -num_from_elite)[-num_from_elite:] # Get the index of the N greatest scores
            elite = population[elite_idx]

            # Delete the already selected individuals by the elitism method
            population = np.delete(population, elite_idx, axis=0)
            fitness = np.delete(fitness, elite_idx, axis=0)
        
        
        
        drawn_from_roulette = []
        if (roulette != 0):
            # Select the next fraction by the roulette method
            num_from_roulette = round(roulette * len(population))
            s = np.sum(fitness)

            for _ in range(num_from_roulette):
                r = np.random.randint(0,s)
                t = 0
                for i, score in enumerate(fitness):
                    t = t + score
                    if (t >= r):
                        drawn_from_roulette.append(population[i])   # get rouletted individual
                        population = np.delete(population, i, axis=0)   # can't be chosen more than once
                        fitness = np.delete(fitness, i, axis=0)   # can't be chosen more than once
                        break

            drawn_from_roulette = np.array(drawn_from_roulette)
        

        selected_population = np.append(elite, drawn_from_roulette, axis=0)

        return selected_population


    # Crossover method used: uniform crossover with random crossover mask
    # Input parameters: list of individual solutions; list of maximum individuals
    # Output returned: new population
    def cross_population(self, population, population_size):

        missing_population = []
        num_missing_individuals = population_size - len(population)

        mask = np.random.randint(0, 2, size=population.shape[1])
        # mask example for a problem with 5 weights [0,1,1,0,1] 
        # Note that, here, each weight is the gene of the chromossome (instead of bits)

        for _ in range(num_missing_individuals):
            dad1_idx = np.random.randint(0, len(population))
            dad2_idx = np.random.randint(0, len(population))
            dad1 = population[dad1_idx]
            dad2 = population[dad2_idx]
            son = []

            for i, gene in enumerate(mask):
                if gene == 0:
                    son.append(dad1[i])
                else:
                    son.append(dad2[i])
            
            son = np.array(son)
            missing_population.append(son)

        missing_population = np.array(missing_population)
        new_population = np.append(population, missing_population, axis=0)
        return new_population


    # Input parameters: list of individual solutions;
    # Output returned: new population
    def mutate_population(self, population, mutation_rate, perturbation_range=0.1):

        mutation = mutation_rate*100
        for individual in population:
            rand = np.random.randint(0, 101)
            
            if self.must_mutate(rand, mutation):
                # Mutate a random number of times, a random number of genes
                for _ in range(np.random.randint(0, len(individual))):
                    mutation_gene = np.random.randint(0, len(individual))
                    perturbation = np.random.uniform(low=-perturbation_range, high=perturbation_range)
                    individual[mutation_gene] += perturbation

        return population


    def must_mutate(self, rand, mutation):
        return rand <= mutation




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