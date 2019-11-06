import controller_template as controller_template
import numpy as np
from datetime import datetime
from math import sin

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
    num_actions = 5
    sensors = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    prev_st = [0, 0, 0, 0, 0, 0.1, 0, 0, 0]
    prev_score = -1000
    features = [1, 0, 0, 0, 0, 0]
    featureNames = ["Const", "diffCheckpoint", "riskFrontal", "riskLeft", "riskRight", "centralized"]
    #featureNames = ["Const", "diffCheckpoint", "riskFrontal", "riskLeft", "riskRight"]
    #featureNames =  ["Const", "diffCheckpoint", "outOfCenter", "needForSpeed", "enemyThreat"]
    num_features = len(features)

    def __init__(self, track, evaluate=True, bot_type=None):
        super().__init__(track, evaluate=evaluate, bot_type=bot_type)

    # METHODS YOU NEED TO IMPLEMENT ###################################

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
        diffCheckpoint = 2*(diffCheckpoint + 20)/40 - 1



        # goal to minimize getting on grass.
        # if not on track, adds spoiler constant.
        riskFrontalCollision = (1 - st[ON_TRACK]) * 200 + st[ON_TRACK]*(st[SPEED] / st[DIST_CENTER])
        riskFrontalCollision = 2* (riskFrontalCollision - 0.1)/199.9 - 1  

        
        riskLeftCollision = (1 - st[ON_TRACK]) * 200 + st[ON_TRACK]*(st[SPEED] / st[DIST_LEFT])
        riskLeftCollision = (riskLeftCollision - 0.1) / 209.9


        riskRightCollision = (1 - st[ON_TRACK]) * 200 + st[ON_TRACK]*(st[SPEED] / st[DIST_RIGHT])
        riskRightCollision = (riskRightCollision - 0.1) / 209.9

        
        centralizedPosition = (abs(st[DIST_LEFT] - st[DIST_RIGHT]) * -1) + 99
        # max: 0 * -1 + 99 = 99
        # min: |100 - 1| * -1 + 99 = 0
        centralizedPosition = centralizedPosition / 99




        """
                COMPUTATIONS OF THE THIRD SET OF FEATURES TESTED



        outOfCenter = (st[DIST_LEFT] - st[DIST_RIGHT]) / 100
        # left - right
        # if closer to left margin, negative number -> turn right
        # if on left side GRASS, distLeft will be a positive value X
        # and distRight will be a positive value 100+X
        # yielding a maximum value of -100 or +100
        # therefore divide by 100 to get range [-1,1]
        # the bigger the difference, the sharper the need for turning right or left
        # this can be softened with some log function so that being closer to center
        # doesnt provoke a linear need to turn (sublinear in comparison to being on grass)


        needForSpeed = st[SPEED] / (1+st[DIST_CENTER])
        needForSpeed = 2*(needForSpeed - 10/101)/(200 + 10/101) -1


        if st[ENEMY_NEAR] == 1:
            if abs(sin(st[ENEMY_ANGLE])) < 0.5:
                enemyThreat = abs(sin(st[ENEMY_ANGLE]))*(101 - st[DIST_ENEMY])
            else:
                abs(sin(st[ENEMY_ANGLE])) * (1 + st[DIST_ENEMY])

            enemyThreat = 2* (enemyThreat - 0)/101 - 1
        else:
            enemyThreat = -1


        self.prev_st = st
        return [1, diffCheckpoint, outOfCenter, needForSpeed, enemyThreat]

        """



        
        self.prev_st = st

        return [1, diffCheckpoint, riskFrontalCollision, riskLeftCollision,
                riskRightCollision, centralizedPosition]




    def learn(self, weights, *argv):
        """
        IMPLEMENT YOUR LEARNING METHOD (i.e. YOUR LOCAL SEARCH ALGORITHM) HERE

        HINT: you can call self.run_episode (see controller_template.py) to evaluate a given set of weights
        :param weights: initial weights of the controller (either loaded from a file or generated randomly)
        :return: the best weights found by your learning algorithm, after the learning process is over
        """

        print("\n\n############### STARTING TRAINING ###############\n\n")
        best_weights, best_score = self.local_search(weights, *argv)
        print("\n\n###############   BEST  WEIGHTS   ###############\n\n")
        print(best_weights)
        Controller.save_result(self, best_weights, best_score)
        return best_weights

    def local_search(self, weights):
        raise NotImplementedError("This Method Must Be Implemented")

    def save_result(self, best_weights, best_score):
        f = open(self.alias + "_" + "iter_w", "w")
        f.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S") + "\n")
        f.write(str(Controller.featureNames))
        f.write("\nScore: " + str(best_score) + "\n")
        f.write("\nWeights: " + ','.join([str(i) for i in best_weights]) + "\n\n")
        f.close()

    

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
