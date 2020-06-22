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
    #cam_offset = np.array([0.05755483, 0.0, 0.16810357])
    right_arm_camera_id = env.sim.model.camera_name2id("eye_on_right_wrist")
    left_arm_camera_id = env.sim.model.camera_name2id("eye_on_left_wrist")

    arena_pos = env.env.mujoco_arena.bin_abs
    init_obj_pos = arena_pos + np.array([0.0, -0.5, 0.0])
    float_pos = arena_pos + np.array([0.0, 0.0, 0.3])
    release_pos = arena_pos + np.array([0.0, 0.5, 0.3])

    num_episodes = args.num_episodes
    num_steps = args.num_steps
    test = args.test
    save_num = args.seed

    success_count, failure_count, controller_failure = 0, 0, 0
    for i in range(0, num_episodes):
        print("Reset!")
        pos = init_obj_pos + np.array([0.0, 0.0, 0.3])
        env.env.model.objects[0].set("pos", array_to_string(pos))
        quat = random_quat()
        env.env.model.objects[0].set("quat", array_to_string(quat))

        geoms = env.env.model.objects[0].findall("./geom")
        #r, g, b, a = 1.0 * np.random.random(), 1.0 * np.random.random(), 1.0 * np.random.random(), 1
        r, g, b, a = 0, 1, 0, 1
        for i in range(len(geoms)):
            geoms[i].set("rgba", str(r) + " " + str(g) + " " + str(b) + " " + str(a))

        env.reset_arms(qpos=INIT_ARM_POS)        
        env.reset_sims()

        for step in range(num_steps):
            left_t_pos, right_t_pos = np.array([0.4, 0.6, 1.0]), np.array([0.4, -0.6, 1.0])
            stucked = move_to_pos(env, left_t_pos, right_t_pos, arm='both', level=1.0, render=False)
            if stucked == -1:
                continue

            camera_obs = env.sim.render(
                camera_name="eye_on_right_wrist",
                width=env.camera_width,
                height=env.camera_height,
                depth=env.camera_depth
            )

            phi, theta = 0.0, 0.0
            pnt_hem, rot_mat = get_camera_pos(init_obj_pos, np.array([0, phi, theta]))
            env.sim.data.cam_xpos[right_arm_camera_id] = pnt_hem
            env.sim.data.cam_xmat[right_arm_camera_id] = rot_mat.flatten()

            # CEM best viewpoint
            sel_start = time.time()
            
            print("Try FC-GQ-CNN")
            [result, depths, d_im], rot_c_im, rot_d_im = env.env.gqcnn(arm='right', vis_on=False, num_candidates=20) ## vis_on 

            if isinstance(result, Iterable):
                sample_grasp_idx = np.random.randint(20, size=1)
                result = result[sample_grasp_idx[0]].grasp
            else:
                if result is None:
                    continue
                else:
                    result = result.grasp

            p_x, p_y = result.center
            graspZ = result.depth

            dx, dy = env.env._pixel2pos(p_x, p_y, graspZ, arena_pos=[0, 0, 0])
            dz = graspZ

            right_t_angle = [result.angle, phi, theta]
            right_t_pos = get_target_pos(pnt_hem, [0, phi, theta], [-dx, dy, dz - 0.1])
            stucked = move_to_6Dpos(env, None, None, right_t_pos, right_t_angle, arm='right', level=4.0, render=False)
            if stucked == -1:
                continue

            right_t_pos = get_target_pos(pnt_hem, [0, phi, theta], [-dx, dy, dz + 0.01])
            stucked = move_to_6Dpos(env, None, None, right_t_pos, right_t_angle, arm='right', level=4.0, render=False)
            if stucked == -1:
                continue

            stop_force_gripper(env, None, None, right_t_pos, right_t_angle, arm='right', left_grasp=0.0, right_grasp=1.0, render=False)
            right_t_pos = get_target_pos(pnt_hem, [0, phi, theta], [-dx, dy, dz - 0.3])
            stucked = move_to_6Dpos(env, None, None, right_t_pos, right_t_angle, arm='right', right_grasp=1.0, level=0.1, render=False)

            collision = env._check_contact(arm='right')
            if collision:
                right_t_pos = float_pos
                right_t_angle = np.array([2 * np.pi * np.random.random(), np.pi / 6.0 * np.random.random() + np.pi / 6.0, 0.75 * np.pi])
                move_to_6Dpos(env, None, None, right_t_pos, right_t_angle, arm='right', right_grasp=1.0, level=1.0, render=True)
            else:
                continue

            collision = env._check_contact(arm='right')
            if collision == False:
                print("Drop the object")
                continue

            phi = 0.0
            theta = 0.0
            pnt_hem, camera_rot_mat = get_camera_pos(right_t_pos, np.array([0, phi, theta]))

            # CEM best viewpoint
            sel_start = time.time()
            
            print("Try FC-GQ-CNN")
            rot_c_im, rot_d_im = get_camera_image(env, pnt_hem, camera_rot_mat, arm='left', vis_on=False)
            mask_rot_d_im, _, _ = segmentation_green_object(rot_c_im, rot_d_im, clip=True, vis_on=False)
            #mask_rot_d_im, _, _ = segmentation_object(rot_c_im, rot_d_im, r, g, b, clip=True, vis_on=False)
            #mask_rot_d_im = rot_d_im
            result, depths, d_im = env.policy.evaluate_gqcnn(rot_c_im, mask_rot_d_im, vis_on=False, num_candidates=30)
            grasp_candidates = []
            dists = []

            '''for i in range(30):
                grasp = result[i].grasp
                p_x, p_y = grasp.center
                graspZ = grasp.depth
                objectZ = mask_rot_d_im[p_y, p_x]
                dx, dy = env.env._pixel2pos(p_x, p_y, graspZ, arena_pos=[0, 0, 0])

                dist = np.sqrt(dx ** 2 + dy ** 2 + (0.6 - graspZ) ** 2)
                if dist > 0.10 and np.abs(graspZ - objectZ) < 0.05:# and np.abs(dx) > 0.05 and np.abs(dy) > 0.05:
                    dists.append(dist)
                    grasp_candidates.append(grasp)'''

            if isinstance(result, Iterable):
                sample_grasp_idx = np.random.randint(10, size=1)
                result = result[sample_grasp_idx[0]].grasp
            else:
                if result is None:
                    continue
                else:
                    result = result.grasp

            p_x, p_y = result.center
            graspZ = result.depth

            dx, dy = env.env._pixel2pos(p_x, p_y, graspZ, arena_pos=[0, 0, 0])
            dz = graspZ

            left_t_angle = [result.angle, phi, theta]

            gqcnn_t_pos = get_target_pos(pnt_hem, [0, phi, theta], [-dx, dy, dz])
            gqcnn_t_angle = np.copy(left_t_angle)

            left_t_pos = get_target_pos(pnt_hem, [0, phi, theta], [-dx, dy, dz - 0.1])
            stucked = move_to_6Dpos(env, left_t_pos, left_t_angle, right_t_pos, right_t_angle, arm='both', right_grasp=1.0, level=4.0, render=render)
            collision = env._check_contact(arm='right')
            if collision == False or stucked == -1:
                print("Failure: drop the object")
                controller_failure += 1
                print('total trial: ', success_count, failure_count, controller_failure)
                continue

            left_t_pos = get_target_pos(pnt_hem, [0, phi, theta], [-dx, dy, dz + 0.01])
            stucked = move_to_6Dpos(env, left_t_pos, left_t_angle, right_t_pos, right_t_angle, arm='both', right_grasp=1.0, level=4.0, render=render)

            collision = env._check_contact(arm='right')
            if collision == False:
                #"Drop the object"
                failure_count += 1
                print('total trial: ', success_count, failure_count, controller_failure)
                continue

            if stucked == -1:
                failure_count += 1
                print('total trial: ', success_count, failure_count, controller_failure)
                continue

            stop_force_gripper(env, left_t_pos, left_t_angle, right_t_pos, right_t_angle, arm='both', left_grasp=1.0, right_grasp=1.0, render=render)
            collision = env._check_contact(arm='left')
            if collision == False:
                failure_count += 1
                print('total trial: ', success_count, failure_count, controller_failure)
                continue

            stop_force_gripper(env, left_t_pos, left_t_angle, right_t_pos, right_t_angle, arm='both', left_grasp=1.0, right_grasp=0.0, render=render)
            right_t_pos = get_target_pos(right_t_pos, right_t_angle, [0.0, 0.0, -0.3])
            move_to_6Dpos(env, left_t_pos, left_t_angle, right_t_pos, right_t_angle, arm='right', left_grasp=1.0, level=0.1, render=render)

            collision = env._check_contact(arm='left')
            if collision:
                success_count += 1
                print('total trial: ', success_count, failure_count, controller_failure)
                continue
            else:
                failure_count += 1
                print('total trial: ', success_count, failure_count, controller_failure)
                continue