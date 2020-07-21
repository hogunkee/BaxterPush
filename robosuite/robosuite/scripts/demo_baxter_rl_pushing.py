import argparse
import numpy as np
import time
from collections.abc import Iterable
import robosuite
import robosuite.utils.transform_utils as T
from robosuite.wrappers import IKWrapper
import matplotlib.pyplot as plt
from robosuite.utils.mjcf_utils import array_to_string, string_to_array
from new_motion_planner import move_to_6Dpos, get_camera_pos, force_gripper, move_to_pos, get_target_pos, get_arm_rotation, object_pass, stop_force_gripper
from cem_vp import select_vp
from grasp_network import VPPNET
import json
from utility import segmentation_green_object, segmentation_object

INIT_ARM_POS = [0.40933302, -1.24377906, 0.68787495, 2.03907987, -0.27229507, 0.8635629,
                0.46484251, 0.12655639, -0.74606415, -0.15337326, 2.04313409, 0.39049096,
                0.30120114, 0.43309788]

from gym import spaces

class BaxterEnv():
    def __init__(self, env, task='push', render=True, using_feature=False, random_spawn=True):
        self.env = env
        self.task = task # 'reach', 'push' or 'pick'
        if task=='reach':
            action_size = 10 #8
        elif task=='push':
            action_size = 12
        elif task=='pick':
            action_size = 12
        self.mov_dist = 0.04 #0.03
        self.action_space = spaces.Discrete(action_size)
        self.action_size = action_size
        self.state = None
        self.grasp = None
        self.init_obj_pos = None
        self.obj_pos = None
        self.target_pos = None

        self.random_spawn = random_spawn
        self.render = render
        self.using_feature = using_feature
        self.max_step = 100

        self.min_reach_dist = None  # for 'reach' task

        self.global_done = False


    def reset(self):
        self.step_count = 0
        arena_pos = self.env.env.mujoco_arena.bin_abs
        self.state = np.array([0.4, 0.6, 1.0, 0.0, 0.0, 0.0, 0.4, -0.6, 1.0, 0.0, 0.0, 0.0])
        self.grasp = 0.0

        self.env.reset()
        if self.random_spawn:
            init_pos = arena_pos + np.array([0.2, 0.2, 0.0]) * np.random.uniform(low=-1.0, high=1.0, size=3) + np.array([0.0, 0.0, 0.1])
        else:
            init_pos = arena_pos + np.array([0.15, 0.10, 0.0]) + np.array([0.0, 0.0, 0.1])

        self.env.model.worldbody.find("./body[@name='CustomObject_0']").set("pos", array_to_string(init_pos))
        self.env.model.worldbody.find("./body[@name='CustomObject_0']").set("quat", array_to_string(np.array([0.0, 0.0, 0.0, 1.0])))
        self.state[6:9] = arena_pos + np.array([0.0, 0.0, 0.16]) #0.06

        if self.random_spawn:
            self.goal = arena_pos + np.array([0.2, 0.2, 0.0]) * np.random.uniform(low=-1.0, high=1.0, size=3) + np.array([0.0, 0.0, 0.1])  # 0.025
            while np.linalg.norm(self.goal[0:2] - init_pos[0:2]) < 0.3:  # <0.08
                self.goal = arena_pos + np.array([0.2, 0.2, 0.0]) * \
                            np.random.uniform(low=-1.0, high=1.0,size=3) + np.array([0.0, 0.0, 0.1])  # 0.025
        else:
            self.goal = arena_pos + np.array([-0.05, -0.15, 0.0]) + np.array([0.0, 0.0, 0.1])  # 0.025

        self.env.model.worldbody.find("./body[@name='CustomObject_1']").set("pos", array_to_string(self.goal))
        self.env.model.worldbody.find("./body[@name='CustomObject_1']").set("quat", array_to_string(
            np.array([0.0, 0.0, 0.0, 1.0])))
        target = self.env.model.worldbody.find("./body[@name='target']")
        target.find("./geom[@name='target']").set("rgba", "0 0 0 0")

        if self.task == 'push' or self.task == 'pick':
            self.state[6:9] = init_pos + np.array([self.mov_dist/2, self.mov_dist/2, 0.0]) * np.random.uniform(low=-1.0, high=1.0, size=3) + np.array([0., 0., 0.06])

        self.env.reset_sims()
        # self.env.reset_arms(qpos=INIT_ARM_POS)

        # print('Block init positions:')
        # print(init_pos)
        # print(self.goal)
        # print()

        stucked = move_to_pos(self.env, [0.4, 0.6, 1.0], [0.4, -0.6, 1.0], arm='both', level=1.0, render=self.render)
        stucked = move_to_6Dpos(self.env, self.state[0:3], self.state[3:6], self.state[6:9], self.state[9:12], arm='both', left_grasp=0.0, right_grasp=self.grasp, level=1.0, render=self.render)

        self.obj_id = self.env.obj_body_id['CustomObject_0']
        self.init_obj_pos = np.copy(self.env.sim.data.body_xpos[self.obj_id])
        self.obj_pos = np.copy(self.env.sim.data.body_xpos[self.obj_id])
        self.pre_obj_pos = self.obj_pos

        self.target_id = self.env.obj_body_id['CustomObject_1']
        self.target_pos = np.copy(self.env.sim.data.body_xpos[self.target_id])
        self.pre_vec = self.target_pos - self.obj_pos
        self.pre_target_pos = self.target_pos

        # self.state[6:9] = self.env._r_eef_xpos
        self.arm_pos = self.env._r_eef_xpos
        self.pre_arm_pos = self.arm_pos
        self.global_done = False

        if self.task == 'reach':
            self.min_reach_dist = np.linalg.norm(self.arm_pos[:2] - self.obj_pos[:2])

        if self.using_feature:
            return np.concatenate([self.state[6:9], self.obj_pos, self.target_pos], axis=0)
        else:
            im_1, im_2 = self.get_camera_obs()
            ## visualizing observations ##
            # fig, ax = plt.subplots(1, 2)
            # ax[0].imshow(im_1)
            # ax[1].imshow(im_2)
            # plt.show()
            return [im_1, im_2]


    def step(self, action):
        # 1 0
        # 8
        # gripper open and close
        self.step_count += 1
        action = action[0][0]
        mov_dist = self.mov_dist

        self.pre_arm_pos = self.arm_pos.copy()
        if action < 8:
            mov_degree = action * np.pi / 4.0
            self.arm_pos = self.arm_pos + np.array([mov_dist * np.cos(mov_degree), mov_dist * np.sin(mov_degree), 0.0])
        elif action == 8:
            self.arm_pos = self.arm_pos + np.array([0.0, 0.0, mov_dist])
            # self.state[6:9] = self.state[6:9] + np.array([0.0, 0.0, mov_dist])
        elif action == 9:
            self.arm_pos = self.arm_pos + np.array([0.0, 0.0, -mov_dist])
            # self.state[6:9] = self.state[6:9] + np.array([0.0, 0.0, -mov_dist])
        elif action == 10:
            self.grasp = 1.00
        elif action == 11:
            self.grasp = 0.0

        #obj_id = self.env.obj_body_id['CustomObject_0']
        #obj_pos = self.env.sim.data.body_xpos[obj_id]
        # self.pre_arm_pos = self.env._r_eef_xpos.copy()
        self.pre_obj_pos = self.obj_pos.copy()
        stucked = move_to_6Dpos(self.env, None, None, self.arm_pos, self.state[9:12], arm='right', left_grasp=0.0,
                                right_grasp=self.grasp, level=1.0, render=self.render)
        self.state[6:9] = self.env._r_eef_xpos
        self.arm_pos = self.state[6:9]
        self.obj_pos = self.env.sim.data.body_xpos[self.obj_id]

        # self.obj_pos = np.copy(self.env.sim.data.body_xpos[self.obj_id])
        if self.task == 'reach' or self.task == 'push':
            self.pre_target_pos = self.target_pos
            self.target_pos = self.env.sim.data.body_xpos[self.target_id]
            # self.target_pos = np.copy(self.env.sim.data.body_xpos[self.target_id])

        vec = self.target_pos - self.obj_pos
        # vec = self.goal - self.state[6:9]
        # print('action:', action)
        # print('pre arm pos:', self.pre_arm_pos)
        # print('arm pos:', self.arm_pos)
        # print('')

        done = False
        reward = 0.0
        if self.task == 'reach':
            if stucked == -1 or 1 - np.abs(self.env.env._right_hand_quat[1]) > 0.01:
                reward = -5.0 #0.0 #np.exp(-1.0 * np.min([np.linalg.norm(self.state[6:9]-self.obj_pos), np.linalg.norm(self.state[6:9]-self.target_pos)]))
                done = True
                print('episode done. [STUCKED]')
            else:
                d1 = np.linalg.norm(self.arm_pos - self.obj_pos)
                # d1 = np.linalg.norm(self.arm_pos[:2] - self.obj_pos[:2])

                # if d1 < self.mov_dist / 2:  # or d2 < 0.025:
                if np.linalg.norm(self.obj_pos - self.pre_obj_pos) > 0.005:
                    reward = 100
                    done = True
                    print('episode done. [SUCCESS]')
                elif d1 < self.min_reach_dist - 0.001:
                    self.min_reach_dist = d1
                    reward = 1.0
                elif self.arm_pos[2] > self.env.env.mujoco_arena.bin_abs[2] + 0.18:
                    reward = -0.2
                else:
                    reward = -0.1
                '''
                d1_old = np.linalg.norm(self.pre_arm_pos[:2] - self.pre_obj_pos[:2])
                # d2 = np.linalg.norm(self.arm_pos[:2] - self.target_pos[:2])
                # d2_old = np.linalg.norm(self.pre_arm_pos[:2] - self.pre_target_pos[:2])
                if d1 < self.mov_dist:
                    if d1_old > self.mov_dist:
                        reward = 5
                    if d1 < self.mov_dist/2: # or d2 < 0.025:
                        reward = 100
                        done = True
                        print('episode done. [SUCCESS]')
                elif d1 > self.mov_dist and d1_old < self.mov_dist:
                    reward = -5
                elif d1_old - d1 > self.mov_dist/2: # or d2_old - d2 > 0.02:
                    reward = 1.0
                elif d1 - d1_old > self.mov_dist/2: # or d2 - d2_old> 0.02:
                    reward = -1.0
                else:
                    reward = -0.1
                # if d1 > 0.025 and d2 > 0.025:
                #     step_penalty = 0.1
                #     reward = 10 * np.max([np.exp(-d1) - np.exp(-d1_old), np.exp(-d2) - np.exp(-d2_old)]) - step_penalty  # range: 0~5 - 0.1
                # else:
                #     reward = 100
                #     done = True
                '''

        elif self.task == 'push':
            C1 = 1
            C2 = 100
            if stucked==-1 or 1-np.abs(self.env.env._right_hand_quat[1]) > 0.01:
                # reward = np.exp(-1.0 * np.min([np.linalg.norm(self.state[6:9]-self.obj_pos), np.linalg.norm(self.state[6:9]-self.target_pos)]))
                # reward = - C1 * np.min([np.linalg.norm(self.target_pos - self.state[6:9]), np.linalg.norm(self.obj_pos - self.state[6:9])])
                reward = -10
                done = True
                print('episode done. [STUCKED]')
            else:
                if np.linalg.norm(vec) < 0.10: #0.05
                    reward = 100
                    done = True
                    print('episode done. [SUCCESS]')

                # print('=' * 30)
                # print(self.obj_pos)
                # print(self.pre_obj_pos)
                # print(self.obj_pos - self.pre_obj_pos)
                # print('='*30)
                x = np.linalg.norm(vec)
                x_old = np.linalg.norm(self.pre_vec)
                d1 = np.linalg.norm(self.arm_pos[:2] - self.obj_pos[:2])
                d1_old = np.linalg.norm(self.pre_arm_pos[:2] - self.pre_obj_pos[:2])
                # get away #
                if d1 > 0.4:
                    reward = -5
                    done = True
                elif d1 > 2 * self.mov_dist:
                    reward = -0.5
                # moving distance reward #
                elif x_old - x > 0.01:
                    reward = 2.0 # 100 * (x_old - x)
                # touching reward #
                elif np.linalg.norm(self.obj_pos - self.pre_obj_pos) > 0.01:
                    reward = 1.0
                # step penalty #
                else:
                    reward = 0.0

                # distance_reward = np.exp( 50*(- np.linalg.norm(vec) + np.linalg.norm(self.pre_vec)) ) - 1
                # touching_reward = np.linalg.norm(self.obj_pos - self.pre_obj_pos) + np.linalg.norm(self.target_pos - self.pre_target_pos)
                # step_penalty = 0.0 #0.1
                # reward = C1 * distance_reward + C2 * touching_reward - step_penalty
                self.pre_vec = vec

        elif self.task == 'pick':
            if self.obj_pos[2] - self.init_obj_pos[2] > 0.3:
                done = True
                reward = 100

        if self.step_count >= self.max_step:
            done = True
            print('Episode stopped. (max step)')

        self.global_done = done
        if self.using_feature:
            state = np.concatenate([self.state[6:9], self.obj_pos, self.target_pos], axis=0)
        else:
            im_1, im_2 = self.get_camera_obs()
            state = [im_1, im_2]

        # print('reward:', reward)
        return state, reward, done, {}

    def get_camera_obs(self):
        # GET CAMAERA IMAGE
        camera_obs = self.env.sim.render(
            camera_name="rlview1",
            width=self.env.camera_width,
            height=self.env.camera_height,
            depth=self.env.camera_depth
        )
        rgb, ddd = camera_obs

        extent = self.env.mjpy_model.stat.extent
        near = self.env.mjpy_model.vis.map.znear * extent
        far = self.env.mjpy_model.vis.map.zfar * extent

        im_depth = near / (1 - ddd * (1 - near / far))
        im_rgb = rgb / 255.0
        im_1 = np.concatenate((im_rgb, im_depth[..., np.newaxis]), axis=2)
        im_1 = np.flip(im_1, axis=0)

        camera_obs = self.env.sim.render(
            camera_name="rlview2",
            width=self.env.camera_width,
            height=self.env.camera_height,
            depth=self.env.camera_depth
        )
        rgb, ddd = camera_obs

        extent = self.env.mjpy_model.stat.extent
        near = self.env.mjpy_model.vis.map.znear * extent
        far = self.env.mjpy_model.vis.map.zfar * extent

        im_depth = near / (1 - ddd * (1 - near / far))
        im_rgb = rgb / 255.0
        im_2 = np.concatenate((im_rgb, im_depth[..., np.newaxis]), axis=2)
        im_2 = np.flip(im_2, axis=0)

        crop = self.env.crop
        if crop is not None:
            im_1 = im_1[(self.env.camera_width - crop) // 2:(self.env.camera_width + crop) // 2, \
                  (self.env.camera_height - crop) // 2:(self.env.camera_height + crop) // 2, :]
            im_2 = im_2[(self.env.camera_width - crop) // 2:(self.env.camera_width + crop) // 2, \
                   (self.env.camera_height - crop) // 2:(self.env.camera_height + crop) // 2, :]

        return [im_1, im_2]


def random_quat():
    rand = np.random.rand(3)
    r1 = np.sqrt(1.0 - rand[0])
    r2 = np.sqrt(rand[0])
    pi2 = np.pi * 2.0
    t1 = pi2 * rand[1]
    t2 = pi2 * rand[2]
    return np.array((np.sin(t1) * r1, np.cos(t1) * r1, np.sin(t2) * r2, np.cos(t2) * r2), dtype=np.float32)    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--seed', type=int, default=0)
    parser.add_argument(
        '--num-objects', type=int, default=2)
    parser.add_argument(
        '--num-episodes', type=int, default=10000)
    parser.add_argument(
        '--num-steps', type=int, default=1)
    parser.add_argument(
        '--render', type=bool, default=True)
    parser.add_argument(
        '--bin-type', type=str, default="table") # table, bin, two
    parser.add_argument(
        '--object-type', type=str, default="cube") # T, Tlarge, L, 3DNet, stick, round_T_large
    parser.add_argument(
        '--test', type=bool, default=False)
    parser.add_argument(
        '--config-file', type=str, default="config_example.yaml")
    args = parser.parse_args()

    np.random.seed(args.seed)

    env = robosuite.make(
        "BaxterPush",
        bin_type=args.bin_type,
        object_type=args.object_type,
        ignore_done=True,
        has_renderer=True,
        camera_name="eye_on_right_wrist",
        gripper_visualization=False,
        use_camera_obs=False,
        use_object_obs=False,
        camera_depth=True,
        num_objects=args.num_objects,
        control_freq=100
    )
    env = IKWrapper(env)

    render = args.render

    cam_offset = np.array([0.05, 0, 0.15855])
    #cam_offset = np.array([0.05755483, 0.0, 0.16810357])
    right_arm_camera_id = env.sim.model.camera_name2id("eye_on_right_wrist")
    left_arm_camera_id = env.sim.model.camera_name2id("eye_on_left_wrist")

    arena_pos = env.env.mujoco_arena.bin_abs
    init_pos = arena_pos + np.array([0.0, 0.0, 0.3])
    init_obj_pos = arena_pos + np.array([0.0, 0.0, 0.0])
    float_pos = arena_pos + np.array([0.0, 0.0, 0.3])

    num_episodes = args.num_episodes
    num_steps = args.num_steps
    test = args.test
    save_num = args.seed

    rl_env = BaxterEnv(env, task='pick', render=render)

    success_count, failure_count, controller_failure = 0, 0, 0
    for i in  range(num_episodes):
        state = rl_env.reset()
        done = False
        while not done:
            state, reward, done, _ = rl_env.step(action)
    '''
    for i in range(0, num_episodes):

        rl_env.reset()
        for j in range(12):
            state, reward, done, _ = rl_env.step(j)
    '''
