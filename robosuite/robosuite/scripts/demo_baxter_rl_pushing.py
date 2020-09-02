import argparse
import numpy as np
import time
import robosuite
from robosuite.wrappers import IKWrapper
import matplotlib.pyplot as plt
from robosuite.utils.mjcf_utils import array_to_string, string_to_array
from robosuite.utils.transform_utils import quat2euler
from new_motion_planner import move_to_6Dpos, get_camera_pos, force_gripper, move_to_pos, get_target_pos, get_arm_rotation, object_pass, stop_force_gripper



INIT_ARM_POS = [0.40933302, -1.24377906, 0.68787495, 2.03907987, -0.27229507, 0.8635629,
                0.46484251, 0.12655639, -0.74606415, -0.15337326, 2.04313409, 0.39049096,
                0.30120114, 0.43309788]

from gym import spaces

class BaxterEnv():
    def __init__(self, env, task='push', continuous=False, render=True, using_feature=False, random_spawn=False, rgbd=False, print_on=False, action_type='3D'):
        self.env = env
        self.task = task # 'reach', 'push' or 'pick'
        self.is_continuous = continuous
        self.rgbd = rgbd
        self.print_on = print_on

        self.action_type = action_type #'2D' # or '3D'

        if self.is_continuous:
            if task=='reach':
                action_dim = 3
            elif task=='push':
                action_dim = 6
            elif task=='pick':
                action_dim = 6
            self.action_space = spaces.Box(-1, 1, [action_dim])  # action: [x, y, z, cos_th, sin_th, gripper]
            self.action_dim = action_dim
        else:
            if task=='reach':
                if self.action_type=='2D':
                    action_size = 8
                elif self.action_type=='3D':
                    action_size = 10 #8
            elif task=='push':
                if self.action_type=='2D':
                    action_size = 8
                elif self.action_type=='3D':
                    action_size = 10 #12
            elif task=='pick':
                action_size = 12
            self.action_space = spaces.Discrete(action_size)
            self.action_size = action_size

        self.mov_dist = 0.02 if self.task=='pick' else 0.04
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
        if self.task=='reach' or self.task=='push':
            self.grasp = 1.0
        if self.task=='reach':
            spawn_range = 0.20
            threshold = 0.25
        else:
            spawn_range = 0.15
            threshold = 0.20

        self.env.reset()

        init_pos = arena_pos + np.array([spawn_range, spawn_range, 0.0]) * np.random.uniform(low=-1.0, high=1.0, size=3) + np.array([0.0, 0.0, 0.05]) #0.1
        self.goal = arena_pos + np.array([spawn_range, spawn_range, 0.0]) * np.random.uniform(low=-1.0, high=1.0, size=3) + np.array([0.0, 0.0, 0.05])  # 0.1
        spawn_count = 0
        while np.linalg.norm(self.goal[0:2] - init_pos[0:2]) < threshold:  # <0.15
            spawn_count += 1
            self.goal = arena_pos + np.array([spawn_range, spawn_range, 0.0]) * \
                        np.random.uniform(low=-1.0, high=1.0, size=3) + np.array([0.0, 0.0, 0.05])  # 0.025
            if spawn_count%10 == 0:
                init_pos = arena_pos + np.array([spawn_range, spawn_range, 0.0]) * np.random.uniform(low=-1.0, high=1.0,size=3) + np.array([0.0, 0.0, 0.05])

        main_block = self.env.model.worldbody.find("./body[@name='CustomObject_0']")
        main_block.set("pos", array_to_string(init_pos))
        target_point = self.env.model.worldbody.find("./body[@name='target']")
        if self.env.num_objects==2:
            target_block = self.env.model.worldbody.find("./body[@name='CustomObject_1']")
            target_block.set("pos", array_to_string(self.goal))
            target_point.find("./geom[@name='target']").set("rgba", "0 0 0 0")
            target = target_block
        elif self.env.num_objects==1:
            target_point.set("pos", array_to_string(self.goal))
            target = target_point

        # self.env.model.worldbody.find("./body[@name='CustomObject_0']").set("pos", array_to_string(init_pos))
        # self.env.model.worldbody.find("./body[@name='CustomObject_1']").set("pos", array_to_string(self.goal))
        if self.task=='pick':
            main_block.set("quat", array_to_string(np.array([0.0, 0.0, 0.0, 1.0])))
            # self.env.model.worldbody.find("./body[@name='CustomObject_0']").set("quat", array_to_string(np.array([0.0, 0.0, 0.0, 1.0])))
            # self.env.model.worldbody.find("./body[@name='CustomObject_1']").set("quat", array_to_string(np.array([0.0, 0.0, 0.0, 1.0])))
        else:
            main_block.set("quat", array_to_string(random_quat()))
            # self.env.model.worldbody.find("./body[@name='CustomObject_0']").set("quat", array_to_string(random_quat()))
            # self.env.model.worldbody.find("./body[@name='CustomObject_1']").set("quat", array_to_string(random_quat()))

        # self.env.model.worldbody.find("./body[@name='CustomObject_1']").find("./geom[@name='CustomObject_1']").set("rgba", "0 0 0 0")
        # target = self.env.model.worldbody.find("./body[@name='target']")
        # target.set("pos", array_to_string(self.goal))
        # target.find("./geom[@name='target']").set("rgba", "0 0 0 0")


        self.env.reset_sims()

        self.obj_id = self.env.obj_body_id['CustomObject_0']
        self.init_obj_pos = np.copy(self.env.sim.data.body_xpos[self.obj_id])
        self.obj_pos = np.copy(self.env.sim.data.body_xpos[self.obj_id])
        self.max_height = self.init_obj_pos[2]

        # self.target_id = self.env.obj_body_id['CustomObject_1']
        # self.target_pos = np.copy(self.env.sim.data.body_xpos[self.target_id])
        if self.env.num_objects==2:
            self.target_id = self.env.obj_body_id['CustomObject_1']
            self.target_pos = np.copy(self.env.sim.data.body_xpos[self.target_id])
        elif self.env.num_objects==1:
            self.target_pos = self.goal
        self.pre_vec = self.target_pos - self.obj_pos

        ## set robot init pos ##
        if self.task=='reach':
            if self.action_type=='2D':
                self.state[8] = arena_pos[2] + 0.08 * np.random.uniform(low=0.47, high=1.0)
                if self.random_spawn:
                    self.state[6:8] = arena_pos[:2] + spawn_range * np.random.uniform(low=-1.0, high=1.0, size=2)
                else:
                    self.state[6:8] = arena_pos[:2]

            elif self.action_type=='3D':
                if self.random_spawn:
                    self.state[6:8] = arena_pos[:2] + spawn_range * np.random.uniform(low=-1.0, high=1.0, size=2)
                    self.state[8] = arena_pos[2] + 0.14 + 0.06 * np.random.uniform(low=-1.0, high=1.0)
                else:
                    self.state[6:8] = arena_pos[:2]
                    self.state[8] = arena_pos[2] + 0.14  # 0.16
                # if self.random_spawn:
                #     self.state[6:8] = arena_pos[:2] + spawn_range * np.random.uniform(low=-1.0, high=1.0, size=2)

        elif self.task=='push':
            align_direction = self.pre_vec[:2] / np.linalg.norm(self.pre_vec[:2])
            self.state[6:8] = self.obj_pos[:2] - 0.08 * align_direction

            if self.action_type == '2D':
                if self.random_spawn:
                    self.state[8] = arena_pos[2] + np.random.uniform(low=0.47, high=1.0) * 0.075
                else:
                    self.state[8] = arena_pos[2] + 0.055

            elif self.action_type == '3D':
                if self.random_spawn:
                    self.state[8] = arena_pos[2] + 0.14 + 0.04 * np.random.uniform(low=-1.0, high=1.0)
                else:
                    self.state[8] = arena_pos[2] + 0.14

        elif self.task=='pick':
            if self.random_spawn:
                self.state[6:8] = self.obj_pos[:2] + 2 * self.mov_dist * np.random.uniform(low=-1.0, high=1.0, size=2)
                self.state[8] = arena_pos[2] + 0.14 + 0.04 * np.random.uniform(low=-1.0, high=1.0)
            else:
                self.state[6:8] = self.obj_pos[:2] + self.mov_dist * np.random.uniform(low=-1.0, high=1.0, size=2)
                self.state[8] = arena_pos[2] + 0.14

        ## move robot arm to init pos ##
        _ = move_to_pos(self.env, [0.4, 0.6, 1.0], [0.4, -0.6, 1.0], arm='both', level=1.0, render=self.render)
        if self.task == 'push':
            _ = move_to_6Dpos(self.env, self.state[0:3], self.state[3:6], self.state[6:9] + np.array([0., 0., 0.1]), self.state[9:12],
                                    arm='both', left_grasp=0.0, right_grasp=self.grasp, level=1.0, render=self.render)
        _ = move_to_6Dpos(self.env, self.state[0:3], self.state[3:6], self.state[6:9], self.state[9:12],
                                arm='both', left_grasp=0.0, right_grasp=self.grasp, level=1.0, render=self.render)

        self.pre_obj_pos = np.copy(self.env.sim.data.body_xpos[self.obj_id])
        if self.env.num_objects==2:
            self.pre_target_pos = np.copy(self.env.sim.data.body_xpos[self.target_id])
        elif self.env.num_objects==1:
            self.pre_target_pos = self.goal

        self.arm_pos = self.env._r_eef_xpos
        self.pre_arm_pos = self.arm_pos.copy()
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
        self.step_count += 1
        if np.squeeze(action)==-1:
            im_1, im_2 = self.get_camera_obs()
            state = [im_1, im_2]
            reward = 0.0
            done = True
            return state, reward, done, {}

        if self.is_continuous:
            # [dx, dy, dz] + [cos(theta), sin(theta), grasp]
            action = np.squeeze(action)
            position_change = action[:3] / 20.0
            self.arm_pos = self.arm_pos + position_change

            if self.task == 'push' or self.task == 'pick':
                cos_theta = action[3]
                sin_theta = action[4]
                grasp = (action[5] + 1.)/2.
                theta = np.arctan2(sin_theta, cos_theta)
                self.state[11] = theta
                self.grasp = grasp
        else:
            # 8 directions
            # up / down
            # gripper close / open
            action = np.squeeze(action) #action[0][0]
            assert action < self.action_size
            mov_dist = self.mov_dist

            self.pre_arm_pos = self.arm_pos.copy()
            if action < 8:
                mov_degree = action * np.pi / 4.0
                self.arm_pos = self.arm_pos + np.array([mov_dist * np.cos(mov_degree), mov_dist * np.sin(mov_degree), 0.0])
            elif action == 8:
                self.arm_pos = self.arm_pos + np.array([0.0, 0.0, mov_dist])
            elif action == 9:
                self.arm_pos = self.arm_pos + np.array([0.0, 0.0, -mov_dist])
            elif action == 10:
                self.grasp = 1.0
            elif action == 11:
                self.grasp = 0.0

        #obj_id = self.env.obj_body_id['CustomObject_0']
        #obj_pos = self.env.sim.data.body_xpos[obj_id]
        # self.pre_arm_pos = self.env._r_eef_xpos.copy()
        # self.pre_obj_pos = self.obj_pos.copy()

        ## check the arm pos is in the working area ##
        if self.arm_pos[0] < 0.15 or self.arm_pos[0] > 0.75:
            stucked = -1
        elif self.arm_pos[1] < -0.62 or self.arm_pos[1] > 0.06:
            stucked = -1
        else:
            stucked = move_to_6Dpos(self.env, None, None, self.arm_pos, self.state[9:12], arm='right', left_grasp=0.0,
                                right_grasp=self.grasp, level=1.0, render=self.render)
        self.state[6:9] = self.env._r_eef_xpos
        self.arm_pos = self.state[6:9]
        self.obj_pos = self.env.sim.data.body_xpos[self.obj_id]

        self.pre_target_pos = self.target_pos.copy()
        if self.env.num_objects==2:
            self.target_pos = self.env.sim.data.body_xpos[self.target_id]
        elif self.env.num_objects==1:
            self.target_pos = self.goal
        vec = self.target_pos - self.obj_pos

        done = False
        reward = 0.0
        arm_euler = quat2euler(self.env.env._right_hand_quat)
        if self.task == 'reach':
            # if stucked == -1 or #1 - np.abs(self.env.env._right_hand_quat[1]) > 0.01:
            if stucked == -1 or check_stucked(arm_euler):
                reward = 0.0 #np.exp(-1.0 * np.min([np.linalg.norm(self.state[6:9]-self.obj_pos), np.linalg.norm(self.state[6:9]-self.target_pos)]))
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

                ## sparse reward ##
                # elif d1 < self.min_reach_dist - 0.001:
                #     self.min_reach_dist = d1
                #     reward = 1.0
                # elif self.arm_pos[2] > self.env.env.mujoco_arena.bin_abs[2] + 0.18:
                #     reward = -0.2
                else:
                    pass # reward = -0.1

        elif self.task == 'push':
            if self.action_type=='2D':
                def get_cos(vec1, vec2):
                    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

                vec_target_obj = self.target_pos - self.obj_pos
                vec_obj_arm = self.obj_pos - self.arm_pos
                # if stucked == -1 or #1 - np.abs(self.env.env._right_hand_quat[1]) > 0.01:
                if stucked == -1 or check_stucked(arm_euler):
                    reward = 0.0  # -10
                    done = True
                    print('episode done. [STUCKED]')
                else:
                    x = np.linalg.norm(vec)
                    x_old = np.linalg.norm(self.pre_vec)
                    d1 = np.linalg.norm(self.arm_pos[:2] - self.obj_pos[:2])
                    # d1_old = np.linalg.norm(self.pre_arm_pos[:2] - self.pre_obj_pos[:2])

                    if np.linalg.norm(vec) < 0.10: #0.05
                        reward = 100
                        done = True
                        print('episode done. [SUCCESS]')
                    elif get_cos(vec_target_obj[:2], vec_obj_arm[:2]) < 0 and np.linalg.norm(vec_obj_arm[:2]) > 0.02:
                        reward = 0.0
                        done = True
                    # get away #
                    elif d1 > 0.4:
                        done = True
                        pass # reward = -5
                    elif d1 > 2 * self.mov_dist:
                        pass # reward = -0.5
                    # moving distance reward #
                    elif x_old - x > 0.01:
                        reward = 2.0 # 100 * (x_old - x)
                    # touching reward #
                    elif np.linalg.norm(self.obj_pos - self.pre_obj_pos) > 0.01:
                        pass # reward = 1.0
                    # step penalty #
                    else:
                        pass # reward = 0.0

                    self.pre_vec = vec

            elif self.action_type=='3D':
                # if stucked == -1 or #1 - np.abs(self.env.env._right_hand_quat[1]) > 0.01:
                if stucked == -1 or check_stucked(arm_euler):
                    reward = 0.0 #-10
                    done = True
                    print('episode done. [STUCKED]')
                else:
                    x = np.linalg.norm(vec)
                    x_old = np.linalg.norm(self.pre_vec)
                    d1 = np.linalg.norm(self.arm_pos[:2] - self.obj_pos[:2])
                    d1_old = np.linalg.norm(self.pre_arm_pos[:2] - self.pre_obj_pos[:2])

                    if np.linalg.norm(vec) < 0.10: #0.05
                        reward = 100
                        done = True
                        print('episode done. [SUCCESS]')
                    # get away #
                    elif d1 > 0.4:
                        done = True
                        pass # reward = -5
                    elif d1 > 2 * self.mov_dist:
                        pass # reward = -0.5
                    # moving distance reward #
                    elif x_old - x > 0.01:
                        pass # reward = 2.0 # 100 * (x_old - x)
                    # touching reward #
                    elif np.linalg.norm(self.obj_pos - self.pre_obj_pos) > 0.01:
                        pass # reward = 1.0
                    # step penalty #
                    else:
                        pass # reward = 0.0

                    self.pre_vec = vec

        elif self.task == 'pick':
            if stucked == -1 or check_stucked(arm_euler):
                reward = 0.0
                done = True
                print('episode done. [STUCKED]')
            else:
                # check for picking up the block #
                if self.obj_pos[2] > self.max_height:
                    if np.abs(self.max_height - self.init_obj_pos[2]) < 0.01:
                        reward = 50
                        print('pick success!')
                    self.max_height = self.obj_pos[2]
                # check for placing the block #
                if np.linalg.norm(self.obj_pos[:2] - self.target_pos[:2]) < self.mov_dist/2 and self.obj_pos[2] - self.target_pos[2] < 0.10:
                    reward = 50
                    done = True
                    print('episode done. [SUCCESS]')

        if not done and self.step_count >= self.max_step:
            done = True
            print('Episode stopped. (max step)')

        self.global_done = done
        if self.using_feature:
            state = np.concatenate([self.state[6:9], self.obj_pos, self.target_pos], axis=0)
        else:
            im_1, im_2 = self.get_camera_obs()
            state = [im_1, im_2]

        if self.print_on:
            print('action:', action, '\t/  reward:', reward)
        return state, reward, done, {}

    def get_camera_obs(self):
        # GET CAMERA IMAGE
        # prepare for rendering #
        _ = self.env.sim.render(
            camera_name="birdview",
            width=10, #self.env.camera_width,
            height=10, #self.env.camera_height,
            depth=False #self.env.camera_depth
        )

        # rl view 1
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
        if self.rgbd:
            im_1 = np.concatenate((im_rgb, im_depth[..., np.newaxis]), axis=2)
            im_1 = np.flip(im_1, axis=0)
        else:
            im_1 = np.flip(im_rgb, axis=0)

        # rl view 2
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
        if self.rgbd:
            im_2 = np.concatenate((im_rgb, im_depth[..., np.newaxis]), axis=2)
            im_2 = np.flip(im_2, axis=0)
        else:
            im_2 = np.flip(im_rgb, axis=0)

        crop = self.env.crop
        if crop is not None:
            im_1 = im_1[(self.env.camera_width - crop) // 2:(self.env.camera_width + crop) // 2, \
                  (self.env.camera_height - crop) // 2:(self.env.camera_height + crop) // 2, :]
            im_2 = im_2[(self.env.camera_width - crop) // 2:(self.env.camera_width + crop) // 2, \
                   (self.env.camera_height - crop) // 2:(self.env.camera_height + crop) // 2, :]

        return [im_1, im_2]


def check_stucked(arm_euler):
    # print(np.array(arm_euler) / np.pi)
    check1 = arm_euler[0] % np.pi < 0.02 or np.pi - arm_euler[0] % np.pi < 0.02
    check2 = arm_euler[1] % np.pi < 0.02 or np.pi - arm_euler[1] % np.pi < 0.02
    return not (check1 and check2)

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
