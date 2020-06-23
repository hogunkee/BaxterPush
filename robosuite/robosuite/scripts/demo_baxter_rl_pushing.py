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
    def __init__(self, env, task='push', render=True):
        self.env = env
        self.task = task # 'push' or 'pick'
        self.action_space = spaces.Discrete(12)
        self.state = None
        self.grasp = None
        self.init_obj_pos = None
        self.obj_pos = None
        self.target_pos = None

        self.action_size = 12
        self.render = render

    def reset(self):
        arena_pos = self.env.env.mujoco_arena.bin_abs
        self.state = np.array([0.4, 0.6, 1.0, 0.0, 0.0, 0.0, 0.4, -0.6, 1.0, 0.0, 0.0, 0.0])
        self.grasp = 0.0
        #init_pos = arena_pos + np.array([0.16, 0.16, 0.0]) * np.random.uniform(low=-1.0, high=1.0, size=3) + np.array([0.0, 0.0, 0.1])
        init_pos = arena_pos + np.array([0.08, 0.0, 0.0]) + np.array([0.0, 0.0, 0.1])
        self.env.model.worldbody.find("./body[@name='CustomObject_0']").set("pos", array_to_string(init_pos))
        self.env.model.worldbody.find("./body[@name='CustomObject_0']").set("quat", array_to_string(np.array([0.0, 0.0, 0.0, 1.0])))
        self.state[6:9] = arena_pos + np.array([0.0, 0.0, 0.06])

        #self.goal = np.array([0.4, -0.6, 1.0]) + np.array([0.16, 0.16, 0.0]) * np.random.uniform(low=-1.0, high=1.0, size=3)
        if self.task == 'pick':
            self.goal = init_pos
            target = self.env.model.worldbody.find("./body[@name='target']")
            target.find("./geom[@name='target']").set("rgba", "0 0 0 0")
        elif self.task == 'push':
            self.goal = arena_pos + np.array([0.16, 0.16, 0.0]) * np.random.uniform(low=-1.0, high=1.0, size=3) + np.array([0.0, 0.0, 0.025])
            while np.linalg.norm(self.goal[0:2] - init_pos[0:2]) < 0.08:
                self.goal = arena_pos + np.array([0.16, 0.16, 0.0]) * np.random.uniform(low=-1.0, high=1.0, size=3) + np.array([0.0, 0.0, 0.025])
        # obj_id = self.env.sim.model.body_name2id("target")
        # self.env.model.worldbody.find("./body[@name='target']").set("pos", array_to_string(self.goal))

        self.env.reset_arms(qpos=INIT_ARM_POS)        
        self.env.reset_sims()
        stucked = move_to_pos(self.env, [0.4, 0.6, 1.0], [0.4, -0.6, 1.0], arm='both', level=1.0, render=self.render)
        stucked = move_to_6Dpos(self.env, self.state[0:3], self.state[3:6], self.state[6:9], self.state[9:12], arm='both', left_grasp=0.0, right_grasp=self.grasp, level=1.0, render=self.render)

        self.obj_id = self.env.obj_body_id['CustomObject_0']
        self.init_obj_pos = np.copy(self.env.sim.data.body_xpos[self.obj_id])
        self.obj_pos = np.copy(self.env.sim.data.body_xpos[self.obj_id])

        if self.task == 'push':
            self.target_id = self.env.obj_body_id['CustomObject_1']
            self.target_pos = np.copy(self.env.sim.data.body_xpos[self.target_id])

        self.state[6:9] = self.env._r_eef_xpos


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
        im_rgb = rgb
        im_1 = np.concatenate((im_rgb, im_depth[..., np.newaxis]), axis=2)

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
        im_rgb = rgb
        im_2 = np.concatenate((im_rgb, im_depth[..., np.newaxis]), axis=2)

        return [im_1, im_2]

    def step(self, action):
        # 1 0
        # 8
        # gripper open and close
        mov_degree = action * np.pi / 4.0
        mov_dist = 0.1

        if action == 0:
            self.state[6:9] = self.state[6:9] + np.array([0.0, 0.0, mov_dist])
        if action == 1:
            self.state[6:9] = self.state[6:9] + np.array([0.0, 0.0, -mov_dist])
        elif action > 1 and action < 10:
            mov_degree = (action - 2) * np.pi / 4.0
            self.state[6:9] = self.state[6:9] + np.array([mov_dist * np.cos(mov_degree), mov_dist * np.sin(mov_degree), 0.0])
        elif action == 10:
            self.grasp = 1.00
        elif action == 11:
            self.grasp = 0.0

        #obj_id = self.env.obj_body_id['CustomObject_0']
        #obj_pos = self.env.sim.data.body_xpos[obj_id]
        stucked = move_to_6Dpos(self.env, None, None, self.state[6:9], self.state[9:12], arm='right', left_grasp=0.0, right_grasp=self.grasp, level=1.0, render=self.render)
        self.state[6:9] = self.env._r_eef_xpos
        # obj_id = self.env.obj_body_id['CustomObject_0']
        self.obj_pos = np.copy(self.env.sim.data.body_xpos[self.obj_id])
        if self.task == 'push':
            self.target_pos = np.copy(self.env.sim.data.body_xpos[self.target_id])

        vec = self.target_pos - self.obj_pos
        # vec = self.goal - self.state[6:9]
        if stucked==-1:
            reward = -10
            done = True
        else:
            reward = - np.linalg.norm(vec)

            if self.task == 'push':
                if np.linalg.norm(vec) < 0.05:
                    reward += 100
                    done = True
                else:
                    done = False
            elif self.task == 'pick':
                if self.obj_pos[2] - self.init_obj_pos[2] > 0.3:
                    done = True
                    reward += 100
                else:
                    done = False

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
        im_rgb = rgb
        im_1 = np.concatenate((im_rgb, im_depth[..., np.newaxis]), axis=2)

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
        im_rgb = rgb
        im_2 = np.concatenate((im_rgb, im_depth[..., np.newaxis]), axis=2)

        return [im_1, im_2], reward, done, {}

def get_camera_image(env, camera_pos, camera_rot_mat, arm='right', vis_on=False):

    if arm == 'right':
        camera_id = env.sim.model.camera_name2id("eye_on_right_wrist")
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
