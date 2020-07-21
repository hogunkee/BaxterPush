import os
import sys
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '../..'))
sys.path.append(os.path.join(FILE_PATH, '..', 'scripts'))
from demo_baxter_rl_pushing import *


task = 'reach' #'push'
render = True
using_feature = False #True

screen_width = 64 #96
screen_height = 64 #96
crop = None #64 #None

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU is not efficient here

env = robosuite.make(
    "BaxterPush",
    bin_type='table',
    object_type='cube',
    ignore_done=True,
    has_renderer=True,
    camera_name="eye_on_right_wrist",
    gripper_visualization=False,
    use_camera_obs=False,
    use_object_obs=False,
    camera_depth=True,
    num_objects=2,
    control_freq=100,
    camera_width=screen_width,
    camera_height=screen_height,
    crop=crop
)
env = IKWrapper(env)
env = BaxterEnv(env, task=task, render=render, using_feature=using_feature)

obs = env.reset()
for _ in range(1000):
    action = [[int(input('action? '))]]
    if action[0][0]==-1:
        break
    # action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print("Action: {},  reward: {}".format(action, reward))
    if done:
        obs = env.reset()
        print('Episode restart.')