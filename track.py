"""
This module implements the Track class, which defines all possible racing tracks used in the game

Attributes:
    track_list (list): Global variable that saves all possible Tracks
"""

import os


track_list = []


class Track:
    def __init__(self, binary_img_path: str, display_img_path: str, name: str):
        """
        Class that defines a single track 
        
        :param binary_img_path: Location of the binary track image
        :param display_img_path: Location of the image that will be displayed 
        :param name: Name of the track used for identification/specification in the command line. 
        """
        self._checkpoints = []
        self.obstacles = []
        self.mask_img_path = os.path.abspath(binary_img_path)
        if os.environ.get('OS','') == 'Windows_NT':
            self.display_img_path = display_img_path
        else:
            self.display_img_path = os.path.abspath(display_img_path)
        self.name = name
        self.episode_length = 500

        self._car1_position = None
        self._car2_position = None
        self._angle_of_cars = None
        self._episode_length = None
        self._timeout = None

        track_list.append(self)

    def add_checkpoint(self, pair_of_coordinates: ((float, float), (float, float))) -> None:
        """
        Adds a checkpoint to the track. All tracks require at least 2 checkpoints to be functional.

        :param pair_of_coordinates: Specify the points/coordinates where the checkpoint will be added
        """
        self._checkpoints.append(pair_of_coordinates)

    def add_parked_bot(self, coordinate: (float, float), angle: float) -> None:
        """
        :param coordinate: coordinate of a parked car bot
        :param angle: orientation of parked car (radians)
        """
        self.obstacles.append((coordinate, angle))

    @property
    def checkpoints(self) -> list:
        """
        :return: list of checkpoints in a track
        """
        if len(self._checkpoints) < 2:
            raise(ValueError("Track must have at least 2 checkpoints"))
        return self._checkpoints

    @property
    def car1_position(self) -> (float, float):
        """
        :return: Position of first car
        """
        if self._car1_position is None:
            raise(ValueError("car1_position not assigned"))
        return self._car1_position

    @car1_position.setter
    def car1_position(self, value: (float, float)):
        self._car1_position = value

    @property
    def car2_position(self) -> (float, float):
        """
        :return: Position of second car (this can be a bot or a second player)
        """
        if self._car2_position is None:
            raise(ValueError("car2_position not assigned"))
        return self._car2_position

    @car2_position.setter
    def car2_position(self, value: (float, float)):
        self._car2_position = value

    @property
    def angle_of_cars(self) -> float:
        """
        :return: Angle in radians
        """
        if self._angle_of_cars is None:
            raise(ValueError("angle_of_cars not assigned"))
        return self._angle_of_cars

    @angle_of_cars.setter
    def angle_of_cars(self, value: float):
        self._angle_of_cars = value

    @property
    def episode_length(self) -> int:
        """
        :return: Number of maximum frames an episode will be executed when using this track.
        """
        if self._angle_of_cars is None:
            raise(ValueError("episode_limit not assigned"))
        return self._episode_length

    @episode_length.setter
    def episode_length(self,value: int):
        self._episode_length = value

    @property
    def timeout(self):
        """
        :return: Maximum time the car has to cross each checkpoint
        """
        if self._angle_of_cars is None:
            raise(ValueError("timeout not assigned"))
        return self._timeout

    @timeout.setter
    def timeout(self,value: int):
        self._timeout = value
