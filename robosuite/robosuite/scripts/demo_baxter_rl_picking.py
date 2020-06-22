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
    def __init__(self, env):
        self.env = env
        self.action_space = spaces.Discrete(4)
        self.state = None

    def reset(self):
        arena_pos = env.env.mujoco_arena.bin_abs
        self.state = np.array([0.4, 0.6, 1.0, 0.0, 0.0, 0.0, 0.4, -0.6, 1.0, 0.0, 0.0, 0.0])
        #init_pos = arena_pos + np.array([0.16, 0.16, 0.0]) * np.random.uniform(low=-1.0, high=1.0, size=3) + np.array([0.0, 0.0, 0.1])
        init_pos = arena_pos + np.array([0.08, 0.0, 0.0]) + np.array([0.0, 0.0, 0.1])
        self.env.model.worldbody.find("./body[@name='CustomObject_0']").set("pos", array_to_string(init_pos))
        self.env.model.worldbody.find("./body[@name='CustomObject_0']").set("quat", array_to_string(np.array([0.0, 0.0, 0.0, 1.0])))
        self.state[6:9] = init_pos + np.array([0.0, 0.0, 0.5]) 
        self.goal = init_pos

        self.env.reset_arms(qpos=INIT_ARM_POS)        
        self.env.reset_sims()
        stucked = move_to_pos(self.env, [0.4, 0.6, 1.0], [0.4, -0.6, 1.0], arm='both', level=1.0, render=render)
        stucked = move_to_6Dpos(self.env, self.state[0:3], self.state[3:6], self.state[6:9], self.state[9:12], arm='both', left_grasp=0.0, right_grasp=0.0, level=1.0, render=True)

        '''self.state[6:9] = self.state[6:9] - np.array([0.0, 0.0, 0.18])
        stucked = move_to_6Dpos(self.env, self.state[0:3], self.state[3:6], self.state[6:9], self.state[9:12], arm='both', left_grasp=0.0, right_grasp=0.0, level=1.0, render=True)
        stucked = stop_force_gripper(self.env, self.state[0:3], self.state[3:6], self.state[6:9], self.state[9:12], arm='right', left_grasp=0.0, right_grasp=1.0, level=1.0, render=True)
        self.state[6:9] = self.state[6:9] + np.array([0.0, 0.0, 0.03])
        stucked = move_to_6Dpos(self.env, self.state[0:3], self.state[3:6], self.state[6:9], self.state[9:12], arm='both', left_grasp=0.0, right_grasp=1.0, level=1.0, render=True)'''

        #stucked = move_to_6Dpos(self.env, self.state[0:3], self.state[3:6], self.state[6:9], self.state[9:12], arm='both', left_grasp=0.0, right_grasp=0.0, level=1.0, render=True)

        obj_id = self.env.obj_body_id['CustomObject_0']
        self.state[6:9] = self.env._r_eef_xpos

    def step(self, action):
        mov_dist = 0.05
        if action == 0:
            self.state[6:9] = self.state[6:9] + np.array([0.0, 0.0, -mov_dist])
            right_grasp = 0.0
        elif action == 1:
            self.state[6:9] = self.state[6:9] + np.array([0.0, 0.0, mov_dist])
            right_grasp = 0.0
        elif action == 3:
            right_grasp = 1.0
        else:
            right_grasp = 0.0

        stucked = move_to_6Dpos(self.env, None, None, self.state[6:9], self.state[9:12], arm='right', left_grasp=0.0, right_grasp=right_grasp, level=1.0, render=True)
        #obj_pos = self.env.sim.data.body_xpos[obj_id]
        #obj_id = self.env.obj_body_id['CustomObject_0']
        #self.env.sim.data.body_xpos[obj_id]
        self.state[6:9] = self.env._r_eef_xpos

        vec = self.goal - self.state[6:9]
        reward = - np.linalg.norm(vec)

        obj_id = self.env.obj_body_id['CustomObject_0']
        #self.env.sim.data.body_xpos[obj_id]

        if np.linalg.norm(vec) < 0.05:
            done = True
        else:
            done = False

        '''camera_id = env.sim.model.camera_name2id("rlview1")
        camera_obs = env.sim.render(
            camera_name="eye_on_right_wrist",
            width=env.camera_width,
            height=env.camera_height,
            depth=env.camera_depth
        )
        camera_obs = env.sim.render(
            camera_name="eye_on_right_wrist",
            width=env.camera_width,
            height=env.camera_height,
            depth=env.camera_depth
        )
        rgb, ddd = camera_obs

        extent = env.mjpy_model.stat.extent
        near = env.mjpy_model.vis.map.znear * extent
        far = env.mjpy_model.vis.map.zfar * extent

        im_depth = near / (1 - ddd * (1 - near / far))
        im_rgb = rgb'''

        #plt.imshow(im_rgb)
        #plt.show()

        return np.array(self.state), reward, done, {}

def get_camera_image(env, camera_pos, camera_rot_mat, arm='right', vis_on=False):

    if arm == 'right':
        camera_id = env.sim.model.camera_name2id("eye_on_right_wrist")
        camera_obs = env.sim.render(
            camera_name="eye_on_right_wrist",
            width=env.camera_width,
            height=env.camera_height,
            depth=env.camera_depth
        )
        env.sim.data.cam_xpos[camera_id] = camera_pos
        env.sim.data.cam_xmat[camera_id] = camera_rot_mat.flatten()

        camera_obs = env.sim.render(
            camera_name="eye_on_right_wrist",
            width=env.camera_width,
            height=env.camera_height,
            depth=env.camera_depth
        )
        if env.camera_depth:
            rgb, ddd = camera_obs

    elif arm == 'left':
        camera_id = env.sim.model.camera_name2id("eye_on_left_wrist")
        camera_obs = env.sim.render(
            camera_name="eye_on_left_wrist",
            width=env.camera_width,
            height=env.camera_height,
            depth=env.camera_depth
        )
        env.sim.data.cam_xpos[camera_id] = camera_pos
        env.sim.data.cam_xmat[camera_id] = camera_rot_mat.flatten()

        camera_obs = env.sim.render(
            camera_name="eye_on_left_wrist",
            width=env.camera_width,
            height=env.camera_height,
            depth=env.camera_depth
        )
        if env.camera_depth:
            rgb, ddd = camera_obs

    extent = env.mjpy_model.stat.extent
    near = env.mjpy_model.vis.map.znear * extent
    far = env.mjpy_model.vis.map.zfar * extent

    im_depth = near / (1 - ddd * (1 - near / far))
    im_rgb = rgb
    #im_depth = np.where(vertical_depth_image > 0.25, vertical_depth_image, 1)

    if vis_on:
        plt.imshow(np.flip(im_rgb, axis=0))
        plt.show()

        plt.imshow(np.flip(im_depth, axis=0), cmap='gray')
        plt.show()

    return np.flip(im_rgb, axis=0), np.flip(im_depth, axis=0)

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
        '--num-objects', type=int, default=1)
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

    rl_env = BaxterEnv(env)

    success_count, failure_count, controller_failure = 0, 0, 0
    for i in range(0, num_episodes):

        rl_env.reset()
        for j in range(15):
            state, reward, done, _ = rl_env.step(0)