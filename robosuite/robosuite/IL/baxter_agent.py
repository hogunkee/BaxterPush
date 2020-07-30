import os
import sys
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '../..'))
sys.path.append(os.path.join(FILE_PATH, '..', 'scripts'))
from demo_baxter_rl_pushing import *

import shutil
import time

import datetime
import pickle
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer('render', 1, 'render the screens')
flags.DEFINE_integer('num_episodes', 10000, 'number of episodes')
flags.DEFINE_integer('use_feature', 0, 'using feature-base states or image-base states.')
flags.DEFINE_string('task', 'reach', 'name of task: [ reach / push / pick ]')

flags.DEFINE_integer('save_data', 1, 'save data or not')
flags.DEFINE_integer('max_buff', 10000, 'number of steps saved in one data file.')
FLAGS = flags.FLAGS
using_feature = (FLAGS.use_feature==1)
if using_feature:
    print('This agent will use feature-based states..!!')
else:
    print('This agent will use image-based states..!!')

render = bool(FLAGS.render)
task = FLAGS.task
num_episodes = FLAGS.num_episodes
save_data = bool(FLAGS.save_data)
max_buff = FLAGS.max_buff

# camera resolution
screen_width = 64
screen_height = 64
crop = None

# Path which data will be saved in.
save_name = os.path.join(FILE_PATH, 'data')

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

    if not os.path.exists(save_name):
        os.makedirs(save_name)

    success_log = []
    buff_states = []
    buff_actions = []
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
            # print(step_count, 'steps \t action: ', action, '\t reward: ', reward)
            cumulative_reward += reward
            if reward>=100:
                success_log.append(1)
            else:
                success_log.append(0)

            # recording the trajectories
            if save_data:
                if not os.path.isdir(save_name):
                    os.makedirs(save_name)
                buff_states.append(obs)
                buff_actions.append(action)
                if len(buff_states) >= max_buff:
                    f_list = os.listdir(save_name)
                    num_pickles = len([f for f in f_list if task in f])
                    save_num = num_pickles // 2
                    with open(os.path.join(save_name, task + '_s_%d.pkl'%save_num), 'wb') as f:
                        pickle.dump(np.array(buff_states), f)
                    with open(os.path.join(save_name, task + '_a_%d.pkl'%save_num), 'wb') as f:
                        pickle.dump(np.array(buff_actions), f)
                    print(save_num, '-th file saved.')
                    buff_states, buff_actions = [], []

        print('Episode %d ends.'%(n+1))
        print(step_count, cumulative_reward)


class GreedyAgent():
    def __init__(self, env):
        self.env = env
        self.task = self.env.task
        # self.using_feature = self.env.using_feature
        self.mov_dist = self.env.mov_dist
        self.action_size = self.env.action_size

    def get_action(self):
        mov_dist = self.mov_dist
        if self.task == 'reach':
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

        elif self.task == 'push':
            vec_target_obj = self.env.target_pos - self.env.obj_pos
            vec_obj_arm = self.env.obj_pos - self.env.arm_pos
            mov_vec_list = []
            for a in range(8):
                mov_degree = a * np.pi / 4.0
                mov_vec_list.append(np.array([mov_dist * np.cos(mov_degree), mov_dist * np.sin(mov_degree)]))
                # elif a == 8:
                #     mov_vec_list.append(np.array([0.0, 0.0, mov_dist]))
                # elif a == 9:
                #     mov_vec_list.append(np.array([0.0, 0.0, -mov_dist]))
            mov_cos_list = [self.get_cos(v, vec_obj_arm[:2]) for v in mov_vec_list]

            if self.get_cos(vec_target_obj[:2], vec_obj_arm[:2]) > 0:
                if self.env.arm_pos[2] < 0.65:
                    action = np.argmax(mov_cos_list)
                else:
                    if np.linalg.norm(vec_obj_arm) > 2.0 * mov_dist:
                        action = 9
                    else:
                        next_obj_arm = [vec_obj_arm[:2] - v for v in mov_vec_list]
                        next_cos_list = [self.get_cos(vec_target_obj[:2], w) for w in next_obj_arm]
                        action = np.argmax(next_cos_list)
                        '''
                        best_a = np.argmax(mov_cos_list)
                        mov_cos_list[best_a] = np.min(mov_cos_list)
                        next_best_a = np.argmax(mov_cos_list)
                        if self.get_cos(mov_vec_list[best_a][:2], vec_target_obj[:2]) > 0:
                            action = best_a
                        else:
                            action = next_best_a
                        action = (action + 4) % 8
                        '''
            else:
                if self.env.arm_pos[2] < 0.65:
                    action = 8
                else:
                    next_obj_arm = [vec_obj_arm[:2] - v for v in mov_vec_list]
                    next_cos_list = [self.get_cos(vec_target_obj[:2], w) for w in next_obj_arm]
                    action = np.argmax(next_cos_list)

        return action


    def get_cos(self, vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

if __name__=='__main__':
    main()


