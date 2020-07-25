import os
import sys
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '../..'))
sys.path.append(os.path.join(FILE_PATH, '..', 'scripts'))
from demo_baxter_rl_pushing import *

import shutil
import time

import datetime
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer('render', 1, 'render the screens')
flags.DEFINE_integer('num_episodes', 100, 'number of episodes')
flags.DEFINE_integer('use_feature', 0, 'using feature-base states or image-base states.')
flags.DEFINE_string('task', 'reach', 'name of task: [ reach / push / pick ]')
FLAGS = flags.FLAGS
using_feature = (FLAGS.use_feature==1)
if using_feature:
    print('This agent will use feature-based states..!!')
else:
    print('This agent will use image-based states..!!')

render = bool(FLAGS.render)
task = FLAGS.task
num_episodes = FLAGS.num_episodes

# camera resolution
screen_width = 64
screen_height = 64
crop = None

# Path which data will be saved in.
saving_path = os.path.join(FILE_PATH, 'data')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU is not efficient here

def main():
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

    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    success_log = []
    for n in range(num_episodes):
        obs = env.reset()
        done = False
        cumulative_reward = 0.0
        step_count = 0
        agent = GreedyAgent(env)

        while not done:
            step_count += 1
            action = agent.get_action()
            obs, reward, done, _ = env.step(action)
            print(step_count, 'steps \t action: ', action, '\t reward: ', reward)
            cumulative_reward += reward
            if reward>=100:
                success_log.append(1)
            else:
                success_log.append(0)

        print('Episode %d ends.'%(n+1))
        print(step_count, cumulative_reward)


class GreedyAgent():
    def __init__(self, env):
        self.env = env
        self.task = self.env.task
        self.using_feature = self.env.using_feature
        self.mov_dist = self.env.mov_dist
        self.action_size = self.env.action_size

    def get_action(self):
        mov_dist = self.mov_dist
        predicted_distance_list = []
        for action in range(self.action_size):
            if action < 8:
                mov_degree = action * np.pi / 4.0
                arm_pos = self.env.arm_pos + np.array([mov_dist * np.cos(mov_degree), mov_dist * np.sin(mov_degree), 0.0])
            elif action == 8:
                arm_pos = self.env.arm_pos + np.array([0.0, 0.0, mov_dist])
            elif action == 9:
                arm_pos = self.env.arm_pos + np.array([0.0, 0.0, -mov_dist])

            if arm_pos[2] < 0.57:
                predicted_distance_list.append(np.inf)
            else:
                dist = np.linalg.norm(arm_pos - self.env.obj_pos)
                predicted_distance_list.append(dist)

        action = np.argmin(predicted_distance_list)
        return action

if __name__=='__main__':
    main()


