import argparse
import numpy as np
import time
from collections.abc import Iterable
import robosuite
import robosuite.utils.transform_utils as T
from robosuite.wrappers import IKWrapper
import matplotlib.pyplot as plt
from robosuite.utils.mjcf_utils import array_to_string, string_to_array
from new_motion_planner import move_to_6Dpos, get_camera_pos, force_gripper, move_to_pos, get_target_pos, get_arm_rotation
from cem_vp import select_vp
from grasp_network import VPPNET
import json
from utility import segementation_green_object

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

def get_camera_image(env, camera_pos, app_angle, arm='right', vis_on=False):

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
        '--num-episodes', type=int, default=100)
    parser.add_argument(
        '--num-steps', type=int, default=1)
    parser.add_argument(
        '--render', type=bool, default=True)
    parser.add_argument(
        '--bin-type', type=str, default="two") # table, bin, two
    parser.add_argument(
        '--object-type', type=str, default="Tlarge") # T, Tlarge, L, 3DNet
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
    #cam_offset = np.array([0.05755483, 0.0, 0.16810357])
    right_arm_camera_id = env.sim.model.camera_name2id("eye_on_right_wrist")
    left_arm_camera_id = env.sim.model.camera_name2id("eye_on_left_wrist")

    arena_pos = env.env.mujoco_arena.bin_abs
    #init_cam_pos = arena_pos + np.array([0.0, -0.3, 0.7])
    #init_arm_pos = arena_pos + np.array([0.0, -0.3, 0.7]) - cam_offset
    init_obj_pos = arena_pos + np.array([0.0, -0.5, 0.0])
    #print(init_cam_pos, init_arm_pos, init_obj_pos)
    float_pos = arena_pos + np.array([0.0, 0.0, 0.2])
    release_pos = arena_pos + np.array([0.0, 0.5, 0.2])
    #release_obj_pos = arena_pos + np.array([0.0, 0.5, 0.0])
    #release_pos = env.env.mujoco_arena.bin2_abs
    #release_pos = arena_pos + np.array([0, 0.59, 0.3])

    num_episodes = args.num_episodes
    num_steps = args.num_steps
    test = args.test
    if test:
        config_file = args.config_file
        with open(config_file) as json_file:
            config = json.load(json_file)
        graspnet = VPPNET(config, model_dir=config["model_dir"], is_training=False)

    for i in range(0, num_episodes):
        print("Reset!")
        #item_list, pos_list, quat_list = place_objects_in_bin(env, num_objects=env.num_objects)
        pos = init_obj_pos + np.array([0.0, 0.0, 0.1])
        env.env.model.objects[0].set("pos", array_to_string(pos))
        quat = random_quat()
        env.env.model.objects[0].set("quat", array_to_string(quat))
        env.reset_sims()
        env.env.reset_arms(qpos=INIT_ARM_POS)

        vt_c_im_list, vt_d_im_list, rot_c_im_list, rot_d_im_list = [], [], [], []
        vt_g_pos_list, vt_g_euler_list, rot_g_pos_list, rot_g_euler_list = [], [], [], []
        g_label_list = []
        success_count, failure_count, controller_failure = 0, 0, 0

        for step in range(num_steps):
            stucked = move_to_pos(env, np.array([0.4, 0.6, 1.0]), np.array([0.4, -0.6, 1.0]), arm='both', level=1.0, render=False)
            #stucked = move_to_pos_lrarm(env, np.array([0.4, 0.6, 1.0]), init_arm_pos, level=1.0, render=render)
            if stucked == -1:
                controller_failure += 1
                continue

            #stucked = move_to_6Dpos_larm(env, release_pos, np.array([0.0, 0.0, 0.0]), render=render)
            #print("cur_cam_pos", env.sim.data.cam_xpos[camera_id]) 
            #vt_c_im, vt_d_im = get_vertical_image(env, init_obj_pos, vis_on=False)
            '''if test:
                prediction = graspnet.predict(vt_d_im.reshape((1, env.camera_width, env.camera_height, 1))[:, 80:176, 80:176, :])
                weight = prediction[0]
                max_weight = np.argmax(weight, axis=-1)
                phi = (prediction[1][0, max_weight[0], 0] + 1.0) * np.pi / 10.0
                theta = (prediction[1][0, max_weight[0], 1] + 1.0) * np.pi
            else:
                theta_targ, phi_targ, q_value = select_vp(num_iters=3, num_comp=3, num_seeds=36, num_gmm_samples=24, gmm_reg_covar=0.001, elite_p=0.25, num_candidates=1)
                phi, theta = phi_targ, theta_targ

            camera_obs = env.sim.render(
                camera_name=env.camera_name,
                width=env.camera_width,
                height=env.camera_height,
                depth=env.camera_depth
            )'''
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
            [result, depths, d_im], rot_c_im, rot_d_im = env.env.gqcnn(arm='right', vis_on=False, num_candidates=10) ## vis_on 

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

            t_angle = [result.angle, phi, theta]

            t_pos = get_target_pos(pnt_hem, [0, phi, theta], [-dx, dy, dz - 0.1])
            stucked = move_to_6Dpos(env, None, None, t_pos, t_angle, arm='right', level=4.0, render=False)

            if stucked == -1:
                controller_failure += 1
                continue

            t_pos = get_target_pos(pnt_hem, [0, phi, theta], [-dx, dy, dz + 0.01])
            stucked = move_to_6Dpos(env, None, None, t_pos, t_angle, arm='right', level=4.0, render=False)
            if stucked == -1:
                failure_count += 1
                g_label_list.append(0.0)
                print('Grasping Failed, Success rate: ', success_count / (success_count + failure_count + controller_failure), success_count / (success_count + failure_count), 
                    ', total trial: ', success_count, failure_count, controller_failure)
                continue

            stucked = force_gripper(env, right_grasp=1.0, render=False)
            t_pos = get_target_pos(pnt_hem, [0, phi, theta], [-dx, dy, dz - 0.3])
            stucked = move_to_6Dpos(env, None, None, t_pos, t_angle, arm='right', right_grasp=1.0, level=0.1, render=False)

            collision = env._check_contact(arm='right')
            if collision:
                success_count += 1
                g_label_list.append(1.0)
                print('Grasping Success, Success rate: ', success_count / (success_count + failure_count + controller_failure), success_count / (success_count + failure_count), 
                    ', total trial: ', success_count, failure_count, controller_failure)
                #move_to_pos(env, release_pos, 0.0, 1.0, level=0.1, render=render)
                #force_gripper(env, grasp=0.0, render=render)
                random_orn = np.array([2 * np.pi * np.random.random(), np.pi / 3.0, np.pi * np.random.random()])

                move_to_6Dpos(env, None, None, float_pos, random_orn, arm='right', right_grasp=1.0, level=1.0, render=False)
                #force_gripper(env, right_grasp=1.0, render=render)
                #move_to_pos(env, None, release_pos, arm='right', grasp=1.0, level=0.1, render=render)
                #force_gripper(env, arm='right', grasp=0.0, render=render)
                #move_to_pos(env, None, np.array([0.4, -0.6, 1.0]), arm='right', grasp=0.0, level=1.0, render=render)

            else:
                failure_count += 1
                g_label_list.append(0.0)
                print('Grasping Failed, Success rate: ', success_count/ (success_count + failure_count + controller_failure), success_count / (success_count + failure_count), 
                    ', total trial: ', success_count, failure_count, controller_failure)
                move_to_pos(env, None, np.array([0.4, -0.6, 1.0]), arm='right', level=1.0, render=False)
                continue

            camera_obs = env.sim.render(
                camera_name="eye_on_left_wrist",
                width=env.camera_width,
                height=env.camera_height,
                depth=env.camera_depth
            )

            #theta_targ, phi_targ, q_value = select_vp(num_iters=3, num_comp=3, num_seeds=36, num_gmm_samples=24, gmm_reg_covar=0.001, elite_p=0.25, num_candidates=1)
            #phi, theta = phi_targ, theta_targ

            phi = np.pi / 10.0 * np.random.random()
            theta = np.pi * np.random.random() + np.pi
            #theta_targ, phi_targ, q_value = select_vp(num_iters=3, num_comp=3, num_seeds=36, num_gmm_samples=24, gmm_reg_covar=0.001, elite_p=0.25, num_candidates=1)
            #phi, theta = phi_targ, theta_targ
            #theta, phi, _ = select_vp(env, float_pos, num_iters=3, num_comp=3, num_seeds=36, num_gmm_samples=24, gmm_reg_covar=0.001, elite_p=0.25, num_candidates=1)
            pnt_hem, rot_mat = get_camera_pos(float_pos, np.array([0, phi, theta]))
            env.sim.data.cam_xpos[left_arm_camera_id] = pnt_hem
            env.sim.data.cam_xmat[left_arm_camera_id] = rot_mat.flatten()

            # CEM best viewpoint
            sel_start = time.time()
            
            print("Try FC-GQ-CNN")
            #[result, depths, d_im], rot_c_im, rot_d_im = env.env.gqcnn(arm='left', vis_on=False, num_candidates=10) ## vis_on
            _, rot_c_im, rot_d_im = env.env.gqcnn(arm='left', vis_on=False, num_candidates=10)
            mask_rot_d_im = segementation_green_object(rot_c_im, rot_d_im, clip=True)
            print(np.min(mask_rot_d_im), np.max(mask_rot_d_im))
            result, depths, d_im = env.policy.evaluate_gqcnn(rot_c_im, mask_rot_d_im, vis_on=True, num_candidates=10)

            np.savez("test_" + str(i) + ".npz", rot_c_im, rot_d_im)

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

            t_angle = [result.angle, phi, theta]

            t_pos = get_target_pos(pnt_hem, [0, phi, theta], [-dx, dy, dz - 0.3])
            stucked = move_to_6Dpos(env, t_pos, t_angle, float_pos, random_orn, arm='both', right_grasp=1.0, level=4.0, render=render)
            t_pos = get_target_pos(pnt_hem, [0, phi, theta], [-dx, dy, dz - 0.1])
            stucked = move_to_6Dpos(env, t_pos, t_angle, float_pos, random_orn, arm='both', right_grasp=1.0, level=4.0, render=render)

            if stucked == -1:
                controller_failure += 1
                continue

            t_pos = get_target_pos(pnt_hem, [0, phi, theta], [-dx, dy, dz - 0.05])
            #t_pos = get_target_pos(pnt_hem, [0, phi, theta], [-dx, dy, dz + 0.01])
            stucked = move_to_6Dpos(env, t_pos, t_angle, None, None, arm='left', right_grasp=1.0, level=4.0, render=render)
            #stucked = move_to_6Dpos(env, t_pos, t_angle, float_pos, random_orn, arm='both', right_grasp=1.0, level=4.0, render=render)
            if stucked == -1:
                failure_count += 1
                g_label_list.append(0.0)    
                print('Grasping Failed, Success rate: ', success_count / (success_count + failure_count + controller_failure), success_count / (success_count + failure_count), 
                    ', total trial: ', success_count, failure_count, controller_failure)
                continue

            force_gripper(env, left_grasp=1.0, right_grasp=1.0, render=render)
            force_gripper(env, left_grasp=1.0, right_grasp=0.0, render=render)

            t_pos = get_target_pos(float_pos, random_orn, [0.0, 0.0, -0.3])
            t_angle = [0.0, random_orn[1], random_orn[2]]
            move_to_6Dpos(env, None, None, t_pos, random_orn, arm='right', left_grasp=1.0, level=0.1, render=render)

            t_pos = get_target_pos(pnt_hem, [0, phi, theta], [-dx, dy, dz - 0.3])
            move_to_6Dpos(env, t_pos, t_angle, None, None, arm='left', left_grasp=1.0, level=0.1, render=render)
            move_to_pos(env, np.array([0.4, 0.6, 1.0]), np.array([0.4, -0.6, 1.0]), arm='both', left_grasp=1.0, level=1.0, render=render)

            collision = env._check_contact(arm='left')
            if collision:
                success_count += 1
                g_label_list.append(1.0)
                print('Grasping Success, Success rate: ', success_count / (success_count + failure_count + controller_failure), success_count / (success_count + failure_count), 
                    ', total trial: ', success_count, failure_count, controller_failure)
                #move_to_pos(env, release_pos, 0.0, 1.0, level=0.1, render=render)
                #force_gripper(env, grasp=0.0, render=render)

                #move_to_6Dpos(env, None, None, float_pos, random_orn, arm='right', grasp=1.0, level=1.0, render=render)
                #force_gripper(env, arm='right', grasp=1.0, render=render)
                #move_to_pos(env, release_pos, None, arm='left', left_grasp=1.0, level=0.1, render=render)
                force_gripper(env, left_grasp=0.0, render=render)
                #move_to_pos(env, np.array([0.4, 0.6, 1.0]), None, arm='left', level=1.0, render=render)

            else:
                failure_count += 1
                g_label_list.append(0.0)
                print('Grasping Failed, Success rate: ', success_count/ (success_count + failure_count + controller_failure), success_count / (success_count + failure_count), 
                    ', total trial: ', success_count, failure_count, controller_failure)
                #move_to_pos(env, None, np.array([0.4, 0.6, 1.0]), arm='right', level=1.0, render=render)
                continue