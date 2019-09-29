"""
This module collects command line arguments and prepares everything needed to run the simulator/game

Example:
    To quickly start the game and observe sensor readings:

        $ python AIRacers.py -t track1 play
"""
import os
import argparse
import random
import datetime
import time
import numpy
import pygame
import simulator
from controller1.controller import Controller
from controller2.controller import Controller as Controller2
import tracks_config as track


# Competition score variabels

def play(track_name: str, b_type: str) -> None:
    """
    Launches the simulator in a mode where the player can control each action with the arrow keys.

    :param str track_name: Name of a track, as defined in tracks_config.py
    :param str b_type: String
    :rtype: None
    """
    play_controller = Controller(track_name, bot_type=b_type)
    game_state = play_controller.game_state
    play_controller.sensors = [53, 66, 100, 1, 172.1353274581511, 150, -1, 0, 0]
    while True:
        events = pygame.event.get()

        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    direction = 1
                    feedback = game_state.frame_step(direction)
                    print("sensors  " + str(feedback))
                    print("features " + str(play_controller.compute_features(feedback)))
                    print("score    " + str(play_controller.game_state.car1.score))
                elif event.key == pygame.K_LEFT:
                    direction = 2
                    feedback = game_state.frame_step(direction)
                    print("sensors  " + str(feedback))
                    print("features " + str(play_controller.compute_features(feedback)))
                    print("score    " + str(play_controller.game_state.car1.score))
                elif event.key == pygame.K_UP:
                    direction = 3
                    feedback = game_state.frame_step(direction)
                    print("sensors  " + str(feedback))
                    print("features " + str(play_controller.compute_features(feedback)))
                    print("score    " + str(play_controller.game_state.car1.score))

                elif event.key == pygame.K_DOWN:
                    direction = 4
                    feedback = game_state.frame_step(direction)
                    print("sensors  " + str(feedback))
                    print("features " + str(play_controller.compute_features(feedback)))
                    print("score    " + str(play_controller.game_state.car1.score))

                if event.key == pygame.K_q:
                    exit()
                if event.key == pygame.K_r:
                    game_state.reset()
        pass


def parser() -> (argparse.Namespace, list):
    """
    Parses command line arguments.

    :return: a tuple containing parsed arguments and leftovers
    """
    p = argparse.ArgumentParser(prog='gamename.py')
    mode_p = p.add_subparsers(dest='mode')
    mode_p.required = True
    p.add_argument('-w', nargs=1,
                   help='Specifies the weights\' file path; if not specified, a random vector of weights will be '
                        'generated.\n')
    p.add_argument('-b', nargs=1, choices=['parked_bots', 'dumb_bot', 'safe_bot', 'ninja_bot', 'custom_bot', 'none'],
                   help='Selects bot type')
    p.add_argument('-t', nargs=1,
                   help='Specifies the track you want to select; by default, track1 will be used. '
                        'Check the \'tracks.py\' file to see the available tracks/create new ones.\n')
    mode_p.add_parser('learn',
                      help='Starts %(prog)s in learning mode. This mode does not render the game to your screen, '
                           'resulting in '
                           'faster learning.\n')
    mode_p.add_parser('evaluate',
                      help='Starts %(prog)s in evaluation mode. This mode runs your AI with the weights/parameters '
                           'passed as parameter \n')
    mode_p.add_parser('play',
                      help='Starts %(prog)s in playing mode. You can control each action of the car using the arrow '
                           'keys of your keyboard.\n')
    mode_p.add_parser('comp',
                      help="Starts %(prog)s in competition mode. Place the controller (controller.py) of player one in controller1/ "
                           "and the other player's controller in controller2/. The weights will be loaded from a file called \"weights.txt\" in "
                           "the same folder.\n")

    arguments, leftovers = p.parse_known_args()
    p.parse_args()
    return arguments, leftovers


def comp(a_track: 'Track',weights_1: numpy.ndarray, weights_2: numpy.ndarray, car1_points: int, car2_points: int) -> (int, int):
    """
    Run competition safely

    :param weights_1: weights from controller 1
    :param weights_2: weights from controller 2
    :param car1_points: controller1's score
    :param car2_points: controller2's score

    :return: None
    """
    ctrl1 = Controller(chosen_track, evaluate=False)
    ctrl2 = Controller2(chosen_track, evaluate=False)

    simulator.evaluate = True
    simulation = simulator.Simulation(a_track, 'player2')
    simulation.frame_step(4)
    frame_current = 0
    episode_length = 500
    while frame_current <= episode_length:
        ctrl1.sensors = simulation.car1.sensors
        ctrl2.sensors = simulation.car_bot.sensors

        simulation.car1.car_step(ctrl1.take_action(weights_1))
        simulation.car_bot.car_step(ctrl2.take_action(weights_2))
        simulation.comp_frame_step()

        frame_current += 1

    print("Player 1: " + str(simulation.car1.score))
    print("Player 2: " + str(simulation.car_bot.score))

    if simulation.car1.score > simulation.car_bot.score:
        print("Winner Player 1 received 3pts")
        car1_points += 3
    elif simulation.car1.score == simulation.car_bot.score:
        print("Both Players received 1pt")
        car2_points += 1
        car1_points += 1
    else:
        print("Winner Player 2 received 3pts")
        car2_points += 3

    return car1_points, car2_points


if __name__ == '__main__':

    args, trash = parser()

    # Selects track; by default track1 will be selected
    chosen_track = track.track1
    if args.t is None:
        chosen_track = track.track1
    else:
        for a_track in track.track.track_list:
            if args.t[0] == a_track.name:
                chosen_track = a_track

    # Sets weights
    if args.w is None:
        ctrl_temp = Controller(chosen_track, bot_type=None, evaluate=False)
        fake_sensors = [53, 66, 100, 1, 172.1353274581511, 150, -1, 0, 0]
        features_len = len(ctrl_temp.compute_features(fake_sensors))
        weights = [random.uniform(-1, 1) for i in range(0, features_len * 5)]
    else:
        weights = numpy.loadtxt(args.w[0])

    # Selects Bot Type
    if args.b is None:
        bot_type = None
    elif args.b[0] == 'none':
        bot_type = None
    else:
        bot_type = args.b[0]

    # Starts simulator in play mode
    if str(args.mode) == 'play':
        play(chosen_track, bot_type)
    # Starts simulator in evaluate mode
    elif str(args.mode) == 'evaluate':
        ctrl = Controller(chosen_track, bot_type=bot_type)
        score = ctrl.run_episode(weights)
    # Starts simulator in learn mode and saves the best results in a file
    elif str(args.mode) == 'learn':
        ctrl = Controller(chosen_track, evaluate=False, bot_type=bot_type)
        result = ctrl.learn(weights)
        if not os.path.exists("./params"):
            os.makedirs("./params")
        output = "./params/%s.txt" % datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S')
        print(output)
        numpy.savetxt(output, result)

    elif str(args.mode) == 'comp':
        w_ctrl1 = numpy.loadtxt('controller1/weights.txt')
        w_ctrl2 = numpy.loadtxt('controller2/weights.txt')

        car1_pts = 0
        car2_pts = 0

        for a_track in track.track.track_list:
            print("Starting race in track %s\n" % a_track.name)
            car1_pts, car2_pts = comp(a_track,w_ctrl1, w_ctrl2, car1_pts, car2_pts)
            print("Switching Sides...\n")
            a_track.car1_position, a_track.car2_position = a_track.car2_position, a_track.car1_position
            car1_pts, car2_pts = comp(a_track,w_ctrl1, w_ctrl2, car1_pts, car2_pts)
            a_track.car1_position, a_track.car2_position = a_track.car2_position, a_track.car1_position

        print("Total score player 1: %dpts" % car1_pts)
        print("Total score player 2: %dpts" % car2_pts)

        if car1_pts > car2_pts:
            print("Player 1 is the winner!!!")
        elif car1_pts == car2_pts:
            print("Oh no! It's a tie.")
        else:
            print("Player 2 is the winner!!!")
