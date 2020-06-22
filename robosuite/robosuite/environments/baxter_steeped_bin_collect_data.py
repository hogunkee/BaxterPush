from collections import OrderedDict
import random
import numpy as np
import os

import robosuite.utils.transform_utils as T
from robosuite.utils.mjcf_utils import string_to_array
from robosuite.utils import MujocoPyRenderer
from robosuite.environments.baxter import BaxterEnv

from robosuite.models.arenas import SteepedBinsArena, TableArena, BinsArena, TwoBinsArena
from robosuite.models.objects import ThreeDNet, TObject, LObject, TlargeObject, roundTlargeObject, CustomObject, CustomRandomizedObject
from robosuite.models.robots import Baxter
from robosuite.models.tasks import SteepedBinPickingTask, UniformRandomSampler

from mujoco_py import MjSim
import matplotlib.pyplot as plt
from robosuite.gqcnn.policy_mujoco import GQCNN

class BaxterSteepedBinCollectData(BaxterEnv):
    def __init__(
        self,
        bin_type="table",
        gripper_right="TwoFingerGripper",
        gripper_left="LeftTwoFingerGripper",
        table_full_size=(0.54, 0.54, 0.1), #0.82),
        table_friction=(1, 0.005, 0.0001),
        use_camera_obs=True,
        use_object_obs=True,
        reward_shaping=False,
        placement_initializer=None,
        single_object_mode=0,
        object_type=None,
        gripper_visualization=False,
        use_indicator_object=False,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_collision_mesh=False,
        render_visual_mesh=True,
        control_freq=10,
        horizon=1000,
        ignore_done=False,
        camera_name="agentview",
        camera_height=256,
        camera_width=256,
        camera_depth=True,
        model_ver=2,
        model_name=None,
        num_objects=1,
        scene=1
    ):
        """
        Args:

            gripper_type (str): type of gripper, used to instantiate
                gripper models from gripper factory.

            table_full_size (3-tuple): x, y, and z dimensions of the table.

            table_friction (3-tuple): the three mujoco friction parameters for
                the table.

            use_camera_obs (bool): if True, every observation includes a
                rendered image.

            use_object_obs (bool): if True, include object (cube) information in
                the observation.

            reward_shaping (bool): if True, use dense rewards.

            placement_initializer (ObjectPositionSampler instance): if provided, will
                be used to place objects on every reset, else a UniformRandomSampler
                is used by default.

            single_object_mode (int): specifies which version of the task to do. Note that
                the observations change accordingly.

                0: corresponds to the full task with all types of objects.

                1: corresponds to an easier task with only one type of object initialized
                   on the table with every reset. The type is randomized on every reset.

                2: corresponds to an easier task with only one type of object initialized
                   on the table with every reset. The type is kept constant and will not
                   change between resets.

            object_type (string): if provided, should be one of "milk", "bread", "cereal",
                or "can". Determines which type of object will be spawned on every
                environment reset. Only used if @single_object_mode is 2.

            gripper_visualization (bool): True if using gripper visualization.
                Useful for teleoperation.

            use_indicator_object (bool): if True, sets up an indicator object that 
                is useful for debugging.

            has_renderer (bool): If true, render the simulation state in
                a viewer instead of headless mode.

            has_offscreen_renderer (bool): True if using off-screen rendering.

            render_collision_mesh (bool): True if rendering collision meshes 
                in camera. False otherwise.

            render_visual_mesh (bool): True if rendering visual meshes
                in camera. False otherwise.

            control_freq (float): how many control signals to receive
                in every second. This sets the amount of simulation time
                that passes between every action input.

            horizon (int): Every episode lasts for exactly @horizon timesteps.

            ignore_done (bool): True if never terminating the environment (ignore @horizon).

            camera_name (str): name of camera to be rendered. Must be
                set if @use_camera_obs is True.

            camera_height (int): height of camera frame.

            camera_width (int): width of camera frame.

            camera_depth (bool): True if rendering RGB-D, and RGB otherwise.
        """

        # task settings
        # self.save = False
        self.object_type = object_type
        self.bin_type = bin_type
        self.single_object_mode = single_object_mode

        self.obj_to_use = None
        self.num_objects = num_objects
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction

        # whether to show visual aid about where is the gripper
        self.gripper_visualization = gripper_visualization

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        super().__init__(
            gripper_right=gripper_right,
            gripper_left=gripper_left,
            gripper_visualization=gripper_visualization,
            use_indicator_object=use_indicator_object,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            use_camera_obs=use_camera_obs,
            camera_name=camera_name,
            camera_height=camera_height,
            camera_width=camera_width,
            camera_depth=camera_depth,
        )

        # reward configuration
        self.reward_shaping = reward_shaping

        # information of objects
        self.object_names = list(self.mujoco_objects.keys())
        self.object_site_ids = [
            self.sim.model.site_name2id(ob_name) for ob_name in self.object_names
        ]

        # id of grippers for contact checking, tentatively only for the right arm
        self.right_finger_names = self.gripper_right.contact_geoms()
        self.left_finger_names = self.gripper_left.contact_geoms()

        # self.sim.data.contact # list, geom1, geom2
        self.collision_check_geom_names = self.sim.model._geom_name2id.keys()
        self.collision_check_geom_ids = [
            self.sim.model._geom_name2id[k] for k in self.collision_check_geom_names
        ]

        # gqcnn policy model
        self.policy = GQCNN(model_ver, model_name)

    def _load_model(self):
        super()._load_model()
        self.mujoco_robot.set_base_xpos([0, 0, 0])

        if self.bin_type == "table":
            # load model for table top workspace
            self.mujoco_arena = BinsArena(
                table_full_size=self.table_full_size, table_friction=self.table_friction
            )
        elif self.bin_type == "bin":
            # load model for table top workspace
            self.mujoco_arena = SteepedBinsArena(
                table_full_size=self.table_full_size, table_friction=self.table_friction
            )
        elif self.bin_type == "two":
            self.mujoco_arena = TwoBinsArena(
                table_full_size=self.table_full_size, table_friction=self.table_friction
            )
        '''elif self.bin_type == "bin":
            # load model for table top workspace
            self.mujoco_arena = BinsArena(
                table_full_size=self.table_full_size, table_friction=self.table_friction
            )
        elif self.bin_type == "steeped_bin":
            # load model for table top workspace
            self.mujoco_arena = SteepedBinsArena(
                table_full_size=self.table_full_size, table_friction=self.table_friction
            )'''

        if self.use_indicator_object:
            self.mujoco_arena.add_pos_indicator()

        # The baxter robot has a pedestal, we want to align it with the table
        #self.mujoco_arena.set_origin([0.5, -0.3, 0])   #.5 -.3
        #self.mujoco_arena.set_origin([0.60, -0.4, 0])
        self.mujoco_arena.set_origin([0.6, -0.4, 0.0])
        #self.mujoco_arena.set_origin([0.45 + self.table_full_size[0] / 2, 0, 0])

        num_objects = self.num_objects
        """if self.object_type == "T":
            obj_list = [TObject]
            item_name_list = ['TObject']
        elif self.object_type == "Tlarge":
            obj_list = [TlargeObject]
            item_name_list = ['TlargeObject']
        elif self.object_type == "roundTlarge":
            obj_list = [roundTlargeObject]
            item_name_list = ['roundTlargeObject']
        elif self.object_type == "L":
            obj_list = [LObject]
            item_name_list = ['LObject']
        elif self.object_type == "3DNet":
            obj_list = [ThreeDNet]
            item_name_list = ['3DNet']"""
        #obj_list = [CustomObject]
        #item_name_list = ['CustomObject']
        obj_list = [CustomRandomizedObject]
        item_name_list = ['CustomRandomizedObject']
        self.item_names = []
        choice = np.random.choice(len(obj_list), num_objects)
        self.ob_inits = (np.array(obj_list)[choice.astype(int)]).tolist()
        self.vis_inits = []

        lst = []
        for j in range(len(self.vis_inits)):
            lst.append((str(self.vis_inits[j]), self.vis_inits[j]()))
        self.visual_objects = lst

        lst = []
        for i in range(len(self.ob_inits)):
            '''if self.object_type == "3DNet":
                N = 50#np.random.randint(1367)
                ob = self.ob_inits[i](N)
            else:
                N = 0
                ob = self.ob_inits[i]()'''
            ob = self.ob_inits[i](self.object_type)
            self.item_names.append(str(item_name_list[choice[i]]) + "_" + str(i))
            lst.append((str(item_name_list[choice[i]]) + "_" + str(i), ob))
            #self.item_names.append(str(item_name_list[choice[i]]) + "_" + str(N) + "_" + str(i))
            #lst.append((str(item_name_list[choice[i]]) + "_" + str(N) + "_" + str(i), ob))

        self.mujoco_objects = OrderedDict(lst)
        self.n_objects = len(self.mujoco_objects)

        # task includes arena, robot, and objects of interest
        self.model = SteepedBinPickingTask(
            self.mujoco_arena,
            self.mujoco_robot,
            self.mujoco_objects,
            self.visual_objects,
        )
        #print(self.item_names[0])
        #print(self.model.worldbody)
        #print(self.model.worldbody.find("./body[@name='bin2']").get("pos"))
        #from robosuite.utils.mjcf_utils import array_to_string, string_to_array
        #self.model.worldbody.find("./body[@name='bin2']").set("pos", array_to_string(np.array([1.0, 1.0, 0.6])))
        #print(self.model.worldbody.find("./body[@name='bin2']").get("pos"))
        self.model.place_objects()
        self.model.place_visual()
        self.bin_pos = string_to_array(self.model.bin1_body.get("pos"))
        self.bin_size = self.model.bin_size

    def add_one_object(self):
        self.n_objects += 1

    def clear_objects(self, obj):
        """
        Clears objects with name @obj out of the task space. This is useful
        for supporting task modes with single types of objects, as in
        @self.single_object_mode without changing the model definition.
        """
        for obj_name, obj_mjcf in self.mujoco_objects.items():
            if obj_name == obj:
                continue
            else:
                sim_state = self.sim.get_state()
                sim_state.qpos[self.sim.model.get_joint_qpos_addr(obj_name)[0]] = 10
                self.sim.set_state(sim_state)
                self.sim.forward()

    def _get_reference(self):
        super()._get_reference()
        self.obj_body_id = {}
        self.obj_geom_id = {}

        # tentatively, just for the right arm.
        self.l_finger_geom_ids = [
            self.sim.model.geom_name2id(x) for x in self.gripper_right.left_finger_geoms
        ]
        self.r_finger_geom_ids = [
            self.sim.model.geom_name2id(x) for x in self.gripper_right.right_finger_geoms
        ]

        for i in range(len(self.ob_inits)):
            obj_str = str(self.item_names[i])
            self.obj_body_id[obj_str] = self.sim.model.body_name2id(obj_str)
            try:
                self.obj_geom_id[obj_str] = self.sim.model.geom_name2id(obj_str + '-0')
            except:
                self.obj_geom_id[obj_str] = self.sim.model.geom_name2id(obj_str)
                pass

        # for checking distance to / contact with objects we want to pick up
        self.target_object_body_ids = list(map(int, self.obj_body_id.values()))
        self.contact_with_object_geom_ids = list(map(int, self.obj_geom_id.values()))

        # keep track of which objects are in their corresponding bins
        self.objects_in_bins = np.zeros(len(self.ob_inits))

        # target locations in bin for each object type
        self.target_bin_placements = np.zeros((len(self.ob_inits), 3))
        for j in range(len(self.ob_inits)):
            bin_id = j
            bin_x_low = self.bin_pos[0]
            bin_y_low = self.bin_pos[1]
            if bin_id == 0 or bin_id == 2:
                bin_x_low -= self.bin_size[0] / 2.
            if bin_id < 2:
                bin_y_low -= self.bin_size[1] / 2.
            bin_x_low += self.bin_size[0] / 4.
            bin_y_low += self.bin_size[1] / 4.
            self.target_bin_placements[j, :] = [bin_x_low, bin_y_low, self.bin_pos[2]]

    def reset_arms(self, qpos):
        self.sim.data.qpos[self._ref_joint_pos_indexes] = qpos #self.mujoco_robot.init_qpos

    def reset_objects(self):
        self.reset()
        self.reset_arms(qpos=[ 3.69420030e-01, -1.05420092e+00,  5.95050381e-01,  1.72831564e+00,
                             -3.36817191e-01,  9.99558867e-01,  3.02551839e-01, -5.31032612e-01,
                              3.02654528e-01, -5.18150277e-04, -3.85248116e-02,  5.12075432e-04,
                             -1.57361097e+00, -2.54388405e-01])
        # (qpos=[1.52438652, -0.47634905, -0.09206436, 1.22671905, \
        #         0.1119182, 0.82371787, 0.62111905, -0.52943496, 0.30266186, \
        #         -0.02294977, -0.03850564, 0.02267607, -1.57361152, -0.25690053])

    def reset_sims(self):
        self._destroy_viewer()
        self.mjpy_model = self.model.get_model(mode="mujoco_py")
        self.sim = MjSim(self.mjpy_model)
        self.viewer = MujocoPyRenderer(self.sim)
        self.viewer.viewer._hide_overlay = True

    def _reset_internal(self):
        super()._reset_internal()

        # reset positions of objects, and move objects out of the scene depending on the mode
        # self.model.place_objects()  #### in the base, already implemented
        if self.single_object_mode == 1:
            self.obj_to_use = (random.choice(self.item_names) + "{}").format(0)
            self.clear_objects(self.obj_to_use)
        elif self.single_object_mode == 2:
            self.obj_to_use = (self.item_names[self.object_id] + "{}").format(0)
            self.clear_objects(self.obj_to_use)

    def reward(self, action=None):
        # compute sparse rewards
        self._check_success()
        reward = np.sum(self.objects_in_bins)

        # add in shaped rewards
        if self.reward_shaping:
            staged_rewards = self.staged_rewards()
            reward += max(staged_rewards)
        return reward

    def staged_rewards(self):
        """
        Returns staged rewards based on current physical states.
        Stages consist of reaching, grasping, lifting, and hovering.
        """

        reach_mult = 0.1
        grasp_mult = 0.35
        lift_mult = 0.5
        hover_mult = 0.7

        # filter out objects that are already in the correct bins
        objs_to_reach = []
        geoms_to_grasp = []
        target_bin_placements = []
        for i in range(len(self.ob_inits)):
            if self.objects_in_bins[i]:
                continue
            obj_str = str(self.item_names[i]) + "0"
            objs_to_reach.append(self.obj_body_id[obj_str])
            geoms_to_grasp.append(self.obj_geom_id[obj_str])
            target_bin_placements.append(self.target_bin_placements[i])
        target_bin_placements = np.array(target_bin_placements)

        ### reaching reward governed by distance to closest object ###
        r_reach = 0.
        if len(objs_to_reach):
            # get reaching reward via minimum distance to a target object
            target_object_pos = self.sim.data.body_xpos[objs_to_reach]
            gripper_site_pos = self.sim.data.site_xpos[self.eef_site_id]
            dists = np.linalg.norm(
                target_object_pos - gripper_site_pos.reshape(1, -1), axis=1
            )
            r_reach = (1 - np.tanh(10.0 * min(dists))) * reach_mult

        ### grasping reward for touching any objects of interest ###
        touch_left_finger = False
        touch_right_finger = False
        for i in range(self.sim.data.ncon):
            c = self.sim.data.contact[i]
            if c.geom1 in geoms_to_grasp:
                bin_id = geoms_to_grasp.index(c.geom1)
                if c.geom2 in self.l_finger_geom_ids:
                    touch_left_finger = True
                if c.geom2 in self.r_finger_geom_ids:
                    touch_right_finger = True
            elif c.geom2 in geoms_to_grasp:
                bin_id = geoms_to_grasp.index(c.geom2)
                if c.geom1 in self.l_finger_geom_ids:
                    touch_left_finger = True
                if c.geom1 in self.r_finger_geom_ids:
                    touch_right_finger = True
        has_grasp = touch_left_finger and touch_right_finger
        r_grasp = int(has_grasp) * grasp_mult

        ### lifting reward for picking up an object ###
        r_lift = 0.
        if len(objs_to_reach) and r_grasp > 0.:
            z_target = self.bin_pos[2] + 0.25
            object_z_locs = self.sim.data.body_xpos[objs_to_reach][:, 2]
            z_dists = np.maximum(z_target - object_z_locs, 0.)
            r_lift = grasp_mult + (1 - np.tanh(15.0 * min(z_dists))) * (
                lift_mult - grasp_mult
            )

        ### hover reward for getting object above bin ###
        r_hover = 0.
        if len(objs_to_reach):
            # segment objects into left of the bins and above the bins
            object_xy_locs = self.sim.data.body_xpos[objs_to_reach][:, :2]
            y_check = (
                np.abs(object_xy_locs[:, 1] - target_bin_placements[:, 1])
                < self.bin_size[1] / 4.
            )
            x_check = (
                np.abs(object_xy_locs[:, 0] - target_bin_placements[:, 0])
                < self.bin_size[0] / 4.
            )
            objects_above_bins = np.logical_and(x_check, y_check)
            objects_not_above_bins = np.logical_not(objects_above_bins)
            dists = np.linalg.norm(
                target_bin_placements[:, :2] - object_xy_locs, axis=1
            )
            # objects to the left get r_lift added to hover reward, those on the right get max(r_lift) added (to encourage dropping)
            r_hover_all = np.zeros(len(objs_to_reach))
            r_hover_all[objects_above_bins] = lift_mult + (
                1 - np.tanh(10.0 * dists[objects_above_bins])
            ) * (hover_mult - lift_mult)
            r_hover_all[objects_not_above_bins] = r_lift + (
                1 - np.tanh(10.0 * dists[objects_not_above_bins])
            ) * (hover_mult - lift_mult)
            r_hover = np.max(r_hover_all)

        return r_reach, r_grasp, r_lift, r_hover

    def not_in_bin(self, obj_pos, bin_id):

        bin_x_low = self.bin_pos[0]
        bin_y_low = self.bin_pos[1]
        if bin_id == 0 or bin_id == 2:
            bin_x_low -= self.bin_size[0] / 2
        if bin_id < 2:
            bin_y_low -= self.bin_size[1] / 2

        bin_x_high = bin_x_low + self.bin_size[0] / 2
        bin_y_high = bin_y_low + self.bin_size[1] / 2

        res = True
        if (
            obj_pos[2] > self.bin_pos[2]
            and obj_pos[0] < bin_x_high
            and obj_pos[0] > bin_x_low
            and obj_pos[1] < bin_y_high
            and obj_pos[1] > bin_y_low
            and obj_pos[2] < self.bin_pos[2] + 0.1
        ):
            res = False
        return res

    def show_image(self):
        di = super()._get_observation()
        
        camera_obs = self.sim.render(
            camera_name=self.camera_name,
            width=self.camera_width,
            height=self.camera_height,
            depth=self.camera_depth,
            #device_id=1,
        )
        if self.camera_depth:
            di["image"], di["depth"] = camera_obs
        else:
            di["image"] = camera_obs
        plt.imshow(np.flip(di["image"], axis=0))
        plt.show()

    def get_image(self):
        di = super()._get_observation()
        
        camera_obs = self.sim.render(
            camera_name=self.camera_name,
            width=self.camera_width,
            height=self.camera_height,
            depth=self.camera_depth,
            #device_id=1,
        )
        if self.camera_depth:
            di["image"], ddd = camera_obs
        else:
            di["image"] = camera_obs
        
        extent = self.mjpy_model.stat.extent
        near = self.mjpy_model.vis.map.znear * extent
        far = self.mjpy_model.vis.map.zfar * extent

        di["depth"] = near / (1 - ddd * (1 - near / far))
        di["depth"] = np.where(di["depth"] > 0.25, di["depth"], 1)

        return np.flip(di["image"], axis=0), np.flip(di["depth"], axis=0)

    def gqcnn(self, arm='right', vis_on=False, num_candidates=10):
        di = super()._get_observation()

        if arm == 'right':
            camera_obs = self.sim.render(
                camera_name="eye_on_right_wrist",
                width=self.camera_width,
                height=self.camera_height,
                depth=self.camera_depth
            )
        elif arm == 'left':
            camera_obs = self.sim.render(
                camera_name="eye_on_left_wrist",
                width=self.camera_width,
                height=self.camera_height,
                depth=self.camera_depth
            )

        if self.camera_depth:
            di["image"], ddd = camera_obs
        else:
            di["image"] = camera_obs

        '''while np.min(ddd) > 0.8:
            print("ddd failed")
            camera_obs = self.sim.render(
                camera_name=self.camera_name,
                width=self.camera_width,
                height=self.camera_height,
                depth=self.camera_depth
            )
            di["image"], ddd = camera_obs'''

        extent = self.mjpy_model.stat.extent
        near = self.mjpy_model.vis.map.znear * extent
        far = self.mjpy_model.vis.map.zfar * extent

        di["depth"] = near / (1 - ddd * (1 - near / far))
        di["depth"] = np.where(di["depth"] > 0.25, di["depth"], 1)

        return self.policy.evaluate_gqcnn(np.flip(di["image"], axis=0), np.flip(di["depth"], axis=0), vis_on=vis_on, num_candidates=num_candidates), np.flip(di["image"], axis=0), np.flip(di["depth"], axis=0)
        
        #return self.policy.evaluate_gqcnn(np.flip(di["image"], axis=0), \
        #        np.flip(di["depth"], axis=0), vis_on=vis_on, num_candidates=num_candidates)

    def _get_right_arm_pos(self):
        return self.sim.data.site_xpos[self.right_eef_site_id]

    def _get_right_arm_quat(self):
        return T.convert_quat(self.sim.data.get_body_xquat("right_hand"), to="xyzw")

    def _get_observation(self):
        """
        Returns an OrderedDict containing observations [(name_string, np.array), ...].
        
        Important keys:
            robot-state: contains robot-centric information.
            object-state: requires @self.use_object_obs to be True.
                contains object-centric information.
            image: requires @self.use_camera_obs to be True.
                contains a rendered frame from the simulation.
            depth: requires @self.use_camera_obs and @self.camera_depth to be True.
                contains a rendered depth map from the simulation
        """
        di = super()._get_observation()
        if self.use_camera_obs:
            camera_obs = self.sim.render(
                camera_name=self.camera_name,
                width=self.camera_width,
                height=self.camera_height,
                depth=self.camera_depth,
            )
            if self.camera_depth:
                di["image"], di["depth"] = camera_obs
            else:
                di["image"] = camera_obs


        # low-level object information
        if self.use_object_obs:

            # remember the keys to collect into object info
            object_state_keys = []

            # for conversion to relative gripper frame
            gripper_pose = T.pose2mat((di["right_eef_pos"], di["right_eef_quat"]))
            world_pose_in_gripper = T.pose_inv(gripper_pose)

            for i in range(len(self.item_names)):

                if self.single_object_mode == 2 and self.object_id != i:
                    # Skip adding to observations
                    continue

                obj_str = str(self.item_names[i])
                obj_pos = self.sim.data.body_xpos[self.obj_body_id[obj_str]]
                obj_quat = T.convert_quat(
                    self.sim.data.body_xquat[self.obj_body_id[obj_str]], to="xyzw"
                )
                di["{}_pos".format(obj_str)] = obj_pos
                di["{}_quat".format(obj_str)] = obj_quat

                # get relative pose of object in gripper frame
                object_pose = T.pose2mat((obj_pos, obj_quat))
                rel_pose = T.pose_in_A_to_pose_in_B(object_pose, world_pose_in_gripper)
                rel_pos, rel_quat = T.mat2pose(rel_pose)
                di["{}_to_eef_pos".format(obj_str)] = rel_pos
                di["{}_to_eef_quat".format(obj_str)] = rel_quat

                object_state_keys.append("{}_pos".format(obj_str))
                object_state_keys.append("{}_quat".format(obj_str))
                object_state_keys.append("{}_to_eef_pos".format(obj_str))
                object_state_keys.append("{}_to_eef_quat".format(obj_str))

            if self.single_object_mode == 1:
                # Zero out other objects observations
                for obj_str, obj_mjcf in self.mujoco_objects.items():
                    if obj_str == self.obj_to_use:
                        continue
                    else:
                        di["{}_pos".format(obj_str)] *= 0.0
                        di["{}_quat".format(obj_str)] *= 0.0
                        di["{}_to_eef_pos".format(obj_str)] *= 0.0
                        di["{}_to_eef_quat".format(obj_str)] *= 0.0

            di["object-state"] = np.concatenate([di[k] for k in object_state_keys])

        return di

    def _check_contact(self, arm='right'):
        """
        Returns True if gripper is in contact with an object.
        """
        collision = False
        for contact in self.sim.data.contact[: self.sim.data.ncon]:
            if arm == 'right' and (
                self.sim.model.geom_id2name(contact.geom1) in self.right_finger_names
                or self.sim.model.geom_id2name(contact.geom2) in self.right_finger_names):
                collision = True
                break
            elif arm == 'left' and (
                self.sim.model.geom_id2name(contact.geom1) in self.left_finger_names
                or self.sim.model.geom_id2name(contact.geom2) in self.left_finger_names):
                collision = True
                break
        return collision

    def _check_success(self):
        """
        Returns True if task has been completed.
        """

        # remember objects that are in the correct bins
        gripper_site_pos = self.sim.data.site_xpos[self.eef_site_id]
        for i in range(len(self.ob_inits)):
            obj_str = str(self.item_names[i])
            obj_pos = self.sim.data.body_xpos[self.obj_body_id[obj_str]]
            dist = np.linalg.norm(gripper_site_pos - obj_pos)
            r_reach = 1 - np.tanh(10.0 * dist)
            self.objects_in_bins[i] = int(
                (not self.not_in_bin(obj_pos, i)) and r_reach < 0.6
            )

        # returns True if a single object is in the correct bin
        if self.single_object_mode == 1 or self.single_object_mode == 2:
            return np.sum(self.objects_in_bins) > 0

        # returns True if all objects are in correct bins
        return np.sum(self.objects_in_bins) == len(self.ob_inits)

    def _gripper_visualization(self):
        """
        Do any needed visualization here. Overrides superclass implementations.
        """
        # color the gripper site appropriately based on distance to nearest object
        if self.gripper_visualization:
            # find closest object
            square_dist = lambda x: np.sum(
                np.square(x - self.sim.data.get_site_xpos("grip_site"))
            )
            dists = np.array(list(map(square_dist, self.sim.data.site_xpos)))
            dists[self.eef_site_id] = np.inf  # make sure we don't pick the same site
            dists[self.eef_cylinder_id] = np.inf
            ob_dists = dists[
                self.object_site_ids
            ]  # filter out object sites we care about
            min_dist = np.min(ob_dists)
            ob_id = np.argmin(ob_dists)
            ob_name = self.object_names[ob_id]

            # set RGBA for the EEF site here
            max_dist = 0.1
            scaled = (1.0 - min(min_dist / max_dist, 1.)) ** 15
            rgba = np.zeros(4)
            rgba[0] = 1 - scaled
            rgba[1] = scaled
            rgba[3] = 0.5

            self.sim.model.site_rgba[self.eef_site_id] = rgba

    def _pixel2pos  (self, X, Y, depth, arena_pos=[0.7, -0.25, 0.57]):
        fovy = 45 #self.sim.model.cam_fovy[0]
        h, w = 256., 256. 
        f = 0.5 * h / np.tan(fovy * np.pi / 360.)
        A = np.array([[0, f, w / 2], [f, 0, h / 2], [0, 0, 1]])
        A_inv = np.linalg.inv(A)
        x, y, z = np.matmul(A_inv, np.array([X, Y, 1]).transpose())
        x *= depth # 0.52/z
        y *= depth # 0.52/z
        return -x + arena_pos[0], -y + arena_pos[1]

