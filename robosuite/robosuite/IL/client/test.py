from env import CarlaEnv
from observation_utils import CarlaObservationConverter
from action_utils import CarlaActionsConverter

obs_converter = CarlaObservationConverter(h=84, w=84, rel_coord_system=False)
action_converter = CarlaActionsConverter('carla-original')
env = CarlaEnv(obs_converter, action_converter, 0, reward_class_name='CarlaReward')

print('Success!!')