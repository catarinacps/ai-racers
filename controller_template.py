import simulator as simulator


class Controller:
    def __init__(self, track: 'Track', evaluate: bool = True, bot_type: str = None):
        """
            This class creates a new racing simulation and implements the controller of a car.
            :param track: The racing track object to be used in the simulation
            :param evaluate: Sets if GUI is visible or not
            :param bot_type: Selects Bot type
        """
        simulator.evaluate = evaluate
        self.track = track
        self.sensors = []
        self.bot_type = bot_type
        self.episode = 1
        self.track_name = track
        self.episode_length = track.episode_length
        self.game_state = simulator.Simulation(track, bot_type)
        self.best_score = -float('inf')
        self.best_features = []
        pass

    def run_episode(self, parameters: list) -> int:

        self.episode += 1
        self.game_state.reset()
        self.sensors = self.game_state.frame_step(4)
        frame_current = 0
        episode_length = self.episode_length

        while frame_current <= episode_length:
            self.sensors = self.game_state.frame_step(self.take_action(parameters))
            frame_current += 1

        score = self.game_state.car1.score

        return score

    def take_action(self, parameters: list) -> int:
        raise NotImplementedError("This Method Must Be Implemented")

    def compute_features(self, sensors: list) -> list:
        raise NotImplementedError("This Method Must Be Implemented")

    def learn(self, weights):
        raise NotImplementedError("This Method Must Be Implemented")
