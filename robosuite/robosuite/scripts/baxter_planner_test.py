import argparse
import numpy as np
import time
from collections.abc import Iterable
import robosuite
import robosuite.utils.transform_utils as T
from robosuite.wrappers import IKWrapper
import matplotlib.pyplot as plt
from robosuite.utils.mjcf_utils import array_to_string, string_to_array
from new_new_motion_planner import move_to_6Dpos, get_camera_pos, force_gripper, move_to_pos, get_target_pos, get_arm_rotation, object_pass, stop_force_gripper
from cem_vp import select_vp
from grasp_network import VPPNET
import json
#from utility import segmentation_green_object, segmentation_object

INIT_ARM_POS = [0.40933302, -1.24377906, 0.68787495, 2.03907987, -0.27229507, 0.8635629,
                0.46484251, 0.12655639, -0.74606415, -0.15337326, 2.04313409, 0.39049096,
                0.30120114, 0.43309788]

'''def get_vertical_image(env, arena_pos, vis_on=False):
    camera_id = env.sim.model.camera_name2id("eye_on_wrist")
    camera_obs = env.sim.render(
        camera_name=env.camera_name,
        width=env.camera_width,
        height=env.camera_height,
        depth=env.camera_depth
    )
    pnt_hem, rot_mat = get_camera_pos(arena_pos, np.array([0.0, 0.0, 0.0]))
    #pnt_hem = camera_pos
    #rot_mat = np.array([[0.0, -1.0, 0.0],[-1.0, 0.0, 0.0],[0.0, 0.0, -1.0]])
    env.sim.data.cam_xpos[camera_id] = pnt_hem
    env.sim.data.cam_xmat[camera_id] = rot_mat.flatten()
    print("cal_cam_pos", pnt_hem)

    camera_obs = env.sim.render(
        camera_name=env.camera_name,
        width=env.camera_width,
        height=env.camera_height,
        depth=env.camera_depth
    )
    if env.camera_depth:
        vertical_color_image, ddd = camera_obs

    extent = env.mjpy_model.stat.extent
    near = env.mjpy_model.vis.map.znear * extent
    far = env.mjpy_model.vis.map.zfar * extent

    vertical_depth_image = near / (1 - ddd * (1 - near / far))
    vertical_depth_image = np.where(vertical_depth_image > 0.25, vertical_depth_image, 1)

    if vis_on:
        plt.imshow(np.flip(vertical_color_image, axis=0))
        plt.show()

        plt.imshow(np.flip(vertical_depth_image, axis=0), cmap='gray')
        plt.show()

    return np.flip(vertical_color_image, axis=0), np.flip(vertical_depth_image, axis=0)'''

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

'''def get_camera_image(env, camera_pos, app_angle, arm='right', vis_on=False):

    if arm == 'right':
        camera_id = env.sim.model.camera_name2id("eye_on_right_wrist")
        camera_obs = env.sim.render(
            camera_name="eye_on_right_wrist",
            width=env.camera_width,
            height=env.camera_height,
            depth=env.camera_depth
        )
        rot_mat = get_arm_rotation(app_angle)
        env.sim.data.cam_xpos[camera_id] = camera_pos
        env.sim.data.cam_xmat[camera_id] = rot_mat.flatten()

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
        rot_mat = get_arm_rotation(app_angle)
        env.sim.data.cam_xpos[camera_id] = camera_pos
        env.sim.data.cam_xmat[camera_id] = rot_mat.flatten()
        print(camera_pos, rot_mat)

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

    return np.flip(im_rgb, axis=0), np.flip(im_depth, axis=0)'''

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
        '--render', type=bool, default=False)
    parser.add_argument(
        '--bin-type', type=str, default="two") # table, bin, two
    parser.add_argument(
        '--object-type', type=str, default="round_T_large") # T, Tlarge, L, 3DNet, stick, round_T_large
    parser.add_argument(
        '--test', type=bool, default=False)
    parser.add_argument(
        '--config-file', type=str, default="config_example.yaml")
    args = parser.parse_args()

    np.random.seed(args.seed)

    '''if args.bin_type == "table":
        env_name = "BaxterCollectData"
    elif args.bin_type == "bin":
        env_name = "BaxterBinCollectData"
    elif args.bin_type == "steeped_bin":
        env_name = "BaxterSteepedBinCollectData"
    else:
        print("Invalid Bin Type!")'''

    env = robosuite.make(
        "BaxterSteepedBinCollectData",
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
    right_arm_camera_id = env.sim.model.camera_name2id("eye_on_right_wrist")
    left_arm_camera_id = env.sim.model.camera_name2id("eye_on_left_wrist")

    arena_pos = env.env.mujoco_arena.bin_abs
    init_obj_pos = arena_pos + np.array([0.0, -0.5, 0.0])
    float_pos = arena_pos + np.array([0.0, 0.0, 0.3])
    release_pos = arena_pos + np.array([0.0, 0.5, 0.3])

    #env.set_robot_joint_positions([
    #    0.00, -0.55, 0.00, 1.28, 0.00, 0.26, 0.00,
    #    0.00, -0.55, 0.00, 1.28, 0.00, 0.26, 0.00,
    #])
    #env.set_robot_joint_positions(INIT_ARM_POS)

    #z-y-x / yaw pitch roll / psi, theta, phi

    #euler = [0.0, 0.75 * np.pi, 1.5 * np.pi]
    #quat = T.euler2quat(euler)
    tait_brayn = [0.0, 0.25 * np.pi, 1.5 * np.pi]
    quat = T.tait2quat(tait_brayn)

    move_to_6Dpos(env, float_pos, quat, render=True)