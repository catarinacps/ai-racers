from controller import controller
from datetime import datetime


class Controller(controller.Controller):
    def __init__(self, track, evaluate=True, bot_type=None):
        super().__init__(track, evaluate=evaluate, bot_type=bot_type)
        self.algorithm = "None"
        self.alias = "nn"

    def local_search(self, initial_weights):

        print("4th algorithm here!")

        return initial_weights*0, 0

    def save_result(self, weights, score):

        f = open(self.alias + "-" + "best_w", "w+")
        f.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S") + "\r\n")
        f.write(super().featureNames)
        f.write("Score: ", score)
        f.write(weights)
        f.close()
