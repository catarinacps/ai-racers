"""
Use this module to define the racing tracks available in the game, or to create new ones.
"""
import track as track
from math import pi


# TRACK1 BEGINS

track1 = track.Track('assets/track_2.png', 'assets/track_2_textura.png', 'track1')

track1.episode_length = 500
track1.timeout = 30
track1.car1_position = (160, 120)
track1.car2_position = (160, 90)
track1.angle_of_cars = 2*pi


track1.add_checkpoint([(278, 68), (284, 162)])
track1.add_checkpoint([(543, 177), (438, 172)])
track1.add_checkpoint([(427, 328), (349, 392)])
track1.add_checkpoint([(611, 186), (692, 252)])
track1.add_checkpoint([(883, 149), (980, 112)])
track1.add_checkpoint([(835, 342), (740, 334)])
track1.add_checkpoint([(802, 503), (871, 571)])
track1.add_checkpoint([(643., 523), (633, 432)])
track1.add_checkpoint([(457, 692), (459, 594)])
track1.add_checkpoint([(142, 615), (150, 520)])
track1.add_checkpoint([(46, 429), (146, 484)])
track1.add_checkpoint([(180, 351), (280, 370)])


# TRACK1 ENDS


# BABY_PARK BEGINS

baby_park = track.Track('assets/baby_park.png', 'assets/baby_park_textura.png', 'baby_park')

baby_park.episode_length = 500
baby_park.timeout = 30
baby_park.car1_position = (308.67982891788586, 271.57617644235376)
baby_park.car2_position = (308.67982891788586, 231.57617644235376)
baby_park.angle_of_cars = 2*pi - (pi/16)


baby_park.add_checkpoint([(441.1051625641563, 271.6561341001731), (415.2613336278362, 166.55071823782586)])
baby_park.add_checkpoint([(699.5764940803684, 144.4274492589917), (696.9703498893376, 42.46074875115163)])
baby_park.add_checkpoint([(867.5504266698458, 225.0554828206116), (969.0228317305006, 182.5896022668501)])
baby_park.add_checkpoint([(718.7143842588047, 370.0719054776359), (723.9466167152078, 473.94020547105106)])
baby_park.add_checkpoint([(483.77766062797707, 427.6930058157275), (510.7796665619176, 533.2955231888633)])
baby_park.add_checkpoint([(244.81284439972373, 544.8478134831403), (221.275776875401, 644.0950121960279)])
baby_park.add_checkpoint([(142.7653171805199, 496.7199034399432), (52.59500724703348, 550.5187436365408)])
baby_park.add_checkpoint([(109.43016970985042, 269.14011286147985), (166.37622717767763, 358.5445117402318)])


# BABY_PARK ENDS


# INTERLAGOS BEGINS
interlagos = track.Track('assets/interlagos.png', 'assets/interlagos_textura.png', 'interlagos')

# Specifies episode length and timeout
interlagos.episode_length = 1000
interlagos.timeout = 100

# Determines both cars' positions and initial angles/orientations (in radians)
interlagos.car1_position = (805, 700-458-30-24)
interlagos.car2_position = (813, 700-483-30-26)
interlagos.angle_of_cars = 2*pi + (pi/16)

# Adds checkpoints, in the order they must be crossed

interlagos.add_checkpoint([(925, 1000-436-300), (999, 1000-454-300)])
interlagos.add_checkpoint([(888, 1000-370-300), (970, 1000-370-300)])
interlagos.add_checkpoint([(923, 1000-300-300), (999, 1000-300-300)])
interlagos.add_checkpoint([(870, 1000-230-300), (917, 1000-166-300)])
interlagos.add_checkpoint([(605.9352634660668, 528.3218085366615), (625.8090013891796, 603.7474974030124)])
interlagos.add_checkpoint([(275, 1000-30-300), (308, 1000-99-300)])
interlagos.add_checkpoint([(222, 1000-123-300), (295, 1000-136-300)])
interlagos.add_checkpoint([(248-10, 1000-221-300-10), (293, 1000-180-300)])
interlagos.add_checkpoint([(484, 1000-313-300), (527, 1000-252-300)])
interlagos.add_checkpoint([(606, 1000-408-300), (680, 1000-408-300)])
interlagos.add_checkpoint([(563, 1000-459-300), (592, 1000-525-300)])
interlagos.add_checkpoint([(380, 1000-546-300), (430, 1000-494-300)])
interlagos.add_checkpoint([(375, 1000-442-300), (444, 1000-431-300)])
interlagos.add_checkpoint([(340, 1000-371-300), (364, 1000-441-300)])
interlagos.add_checkpoint([(248, 1000-457-300), (293, 1000-509-300)])
interlagos.add_checkpoint([(175, 1000-506-300), (224, 1000-456-300)])
interlagos.add_checkpoint([(150, 1000-413-300), (222, 1000-422-300)])
interlagos.add_checkpoint([(183, 1000-347-300), (257, 1000-340-300)])
interlagos.add_checkpoint([(131, 1000-285-300), (163, 1000-219-300)])
interlagos.add_checkpoint([(35, 1000-232-300), (91, 1000-280-300)])
interlagos.add_checkpoint([(21, 1000-476-300), (92, 1000-453-300)])
interlagos.add_checkpoint([(198, 1000-628-300), (221, 1000-564-300)])
interlagos.add_checkpoint([(444, 1000-594-300), (457, 1000-663-300)])
interlagos.add_checkpoint([(727, 1000-508-300), (753, 1000-592-300)])

# INTERLAGOS ENDS


# MANY_FORKS BEGINS

track1 = track.Track('assets/many_forks.png', 'assets/many_forks_textura.png', 'many_forks')

track1.episode_length = 500
track1.timeout = 30
track1.car1_position = (320, 65)
track1.car2_position = (320, 30)
track1.angle_of_cars = 2*pi


track1.add_checkpoint([(469.18827680684956, 28.343685444907454), (429.0655212812058, 120.14063778936024)])
track1.add_checkpoint([(543.9247381367792, 357.2027235410929), (470.4683809634382, 414.7387996743742)])
track1.add_checkpoint([(626.5931450045899, 351.66784397651924), (718.2750179307252, 359.31206903572877)])
track1.add_checkpoint([(726.1850765427714, 241.80069203508953), (632.7501357793525, 245.08563920895332)])
track1.add_checkpoint([(847.3201466288817, 216.07653210342832), (955.9365208478811, 207.4190644116556)])
track1.add_checkpoint([(977.3633421936397, 510.3469484636485), (881.830271507271, 529.893582793526)])
track1.add_checkpoint([(855.5596550089699, 563.3581607551573), (868.8231651751328, 659.4470742575606)])
track1.add_checkpoint([(508.99447985783434, 550.8357186378623), (505.7708282569279, 685.3311739914011)])
track1.add_checkpoint([(161.71755180964377, 581.5082965536822), (163.61018023949643, 674.48903626695)])
track1.add_checkpoint([(108.78240755748196, 374.0171209554241), (8.18694865375237, 379.7155998564073)])

# MANY_FORKS ENDS










