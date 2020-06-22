import argparse
import numpy as np
import time
from collections.abc import Iterable
import robosuite
import robosuite.utils.transform_utils as T
from robosuite.wrappers import IKWrapper
import matplotlib.pyplot as plt
from robosuite.utils.mjcf_utils import array_to_string, string_to_array
from motion_planner import approach_to_object, move_to_6Dpos, get_camera_pos, force_gripper, move_to_pos_lrarm, move_to_pos, get_pos_and_rot

INIT_ARM_POS = [0.40933302, -1.24377906, 0.68787495, 2.03907987, -0.27229507, 0.8635629,
                0.46484251, 0.12655639, -0.74606415, -0.15337326, 2.04313409, 0.39049096,
                0.30120114, 0.43309788]

def get_vertical_image(env, arena_pos, vis_on=False):
    camera_id = env.sim.model.camera_name2id("eye_on_wrist")
    camera_obs = env.sim.render(
        camera_name=env.camera_name,
        width=env.camera_width,
        height=env.camera_height,
        depth=env.camera_depth
    )
    pnt_hem, rot_mat = get_camera_pos(arena_pos, np.array([0.0, 0.0, 0.0]))
    env.sim.data.cam_xpos[camera_id] = pnt_hem
    env.sim.data.cam_xmat[camera_id] = rot_mat.flatten()

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

    return np.flip(vertical_color_image, axis=0), np.flip(vertical_depth_image, axis=0)

def random_quat(rand=None):
    """Return uniform random unit quaternion.
    rand: array like or None
        Three independent random variables that are uniformly distributed
        between 0 and 1.
    >>> q = random_quat()
    >>> np.allclose(1.0, vector_norm(q))
    True
    >>> q = random_quat(np.random.random(3))
    >>> q.shape
    (4,)
    """
    if rand is None:
        rand = np.random.rand(3)
    else:
        assert len(rand) == 3
    r1 = np.sqrt(1.0 - rand[0])
    r2 = np.sqrt(rand[0])
    pi2 = np.pi * 2.0
    t1 = pi2 * rand[1]
    t2 = pi2 * rand[2]
    return np.array(
        (np.sin(t1) * r1, np.cos(t1) * r1, np.sin(t2) * r2, np.cos(t2) * r2),
        dtype=np.float32,
    )    

def place_objects_in_bin(env, num_objects=10):

    for i in range(num_objects):
        object_z = env.env.model.bin_size[2] + 0.5
        object_x = 0.0
        object_y = 0.0
        """object_z = env.env.model.bin_size[2] + 0.3
        object_x = 0.1 * np.random.random() - 0.05
        object_y = 0.1 * np.random.random() - 0.05"""
        object_xyz = np.array([object_x, object_y, object_z])
        pos = env.env.model.bin_offset + object_xyz
        env.env.model.objects[i].set("pos", array_to_string(pos))
        quat = random_quat()
        env.env.model.objects[i].set("quat", array_to_string(quat))
        env.reset_sims()
        for j in range(0, 500):
            env.env.sim.step()
        for k in range(0, i + 1):
            obj_str = env.env.item_names[k]
            obj_id = env.env.obj_body_id[obj_str]
            pos = env.env.sim.data.body_xpos[obj_id]
            quat = env.env.sim.data.body_xquat[obj_id]
            env.env.model.objects[k].set("pos", array_to_string(pos))
            env.env.model.objects[k].set("quat", array_to_string(quat))
    env.reset_sims()

    bin_pos = env.env.model.bin_offset
    for i in range(num_objects):
        obj_str = env.env.item_names[i]
        obj_id = env.env.obj_body_id[obj_str]
        pos = env.env.sim.data.body_xpos[obj_id]
        print(obj_str, pos)
        if abs(pos[0] - bin_pos[0]) < env.bin_size[0] / 2.0 and abs(pos[1] - bin_pos[1]) < env.bin_size[1] / 2.0:
            continue
        else:
            object_z = env.env.model.bin_size[2] + 0.3
            object_x = 0.1 * np.random.random() - 0.05
            object_y = 0.1 * np.random.random() - 0.05
            object_xyz = np.array([object_x, object_y, object_z])
            pos = bin_pos + object_xyz
            env.env.model.objects[i].set("pos", array_to_string(pos))
            env.reset_sims()
            for j in range(0, 500):
                env.env.sim.step()
            for k in range(0, num_objects):
                obj_str = env.env.item_names[k]
                obj_id = env.env.obj_body_id[obj_str]
                pos = env.env.sim.data.body_xpos[obj_id]
                quat = env.env.sim.data.body_xquat[obj_id]
                env.env.model.objects[k].set("pos", array_to_string(pos))
                env.env.model.objects[k].set("quat", array_to_string(quat))
        env.reset_sims()

    item_list = []
    pos_list = []
    quat_list = []
    for i in range(num_objects):
        obj_str = env.item_names[i]
        obj_id = env.obj_body_id[obj_str]
        pos = env.sim.data.body_xpos[obj_id]
        quat = env.sim.data.body_xquat[obj_id]
        item_list.append(obj_str.split("_")[1])
        pos_list.append(pos)
        quat_list.append(quat)
    return item_list, pos_list, quat_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--seed', type=int, default=0)
    parser.add_argument(
        '--num-objects', type=int, default=5)
    parser.add_argument(
        '--num-episodes', type=int, default=100)
    parser.add_argument(
        '--num-steps', type=int, default=5)
    parser.add_argument(
        '--render', type=bool, default=True)
    parser.add_argument(
        '--bin-type', type=str, default="table") # table, bin
    parser.add_argument(
        '--object-type', type=str, default="T") # T, L, 3DNet
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
        camera_name="eye_on_wrist",
        gripper_visualization=True,
        use_camera_obs=False,
        camera_depth=True,
        num_objects=args.num_objects,
        control_freq=100
    )
    env = IKWrapper(env)
    render = args.render

    cam_offset = np.array([0.05, 0, 0.15855])
    camera_id = env.sim.model.camera_name2id("eye_on_wrist")

    arena_pos = env.env.mujoco_arena.bin_abs
    init_pos = arena_pos + np.array([0, 0, 0.7]) - cam_offset
    release_pos = arena_pos + np.array([0, 0.59, 0.3])

    num_episodes = args.num_episodes
    num_steps = args.num_steps
    for i in range(0, num_episodes):
        print("Reset!")
        item_list, pos_list, quat_list = place_objects_in_bin(env, num_objects=env.num_objects)
        print("Place all objects in bin!")
        vt_c_im_list, vt_d_im_list, rot_c_im_list, rot_d_im_list = [], [], [], []
        vt_g_pos_list, vt_g_euler_list, rot_g_pos_list, rot_g_euler_list = [], [], [], []
        g_label_list = []
        success_count, failure_count, controller_failure = 0, 0, 0

        for step in range(num_steps):
            stucked = move_to_pos_lrarm(env, np.array([0.4, 0.6, 1.0]), np.array([0.4, -0.6, 1.0]), level=1.0, render=render)
            if stucked == -1:
                controller_failure += 1
                continue
            
            vt_c_im, vt_d_im = get_vertical_image(env, arena_pos, vis_on=False)
            phi = np.pi / 6.0 * np.random.random()
            theta = 2 * np.pi * np.random.random()

            camera_obs = env.sim.render(
                camera_name=env.camera_name,
                width=env.camera_width,
                height=env.camera_height,
                depth=env.camera_depth
            )
            pnt_hem, rot_mat = get_camera_pos(arena_pos, np.array([0, phi, theta]))
            env.sim.data.cam_xpos[camera_id] = pnt_hem
            env.sim.data.cam_xmat[camera_id] = rot_mat.flatten()

            # CEM best viewpoint
            sel_start = time.time()
            
            print("Try FC-GQ-CNN")
            [result, depths, d_im], rot_c_im, rot_d_im = env.env.gqcnn(vis_on=False, num_candidates=10) ## vis_on 

            if isinstance(result, Iterable):
                sample_grasp_idx = np.random.randint(10, size=1)
                result = result[sample_grasp_idx[0]].grasp
            else:
                if result is None:
                    continue
                else:
                    result = result.grasp

            p_x, p_y = result.center
            print('p_x, p_y: ', p_x, p_y)

            graspZ = result.depth
            print('graspZ: ', graspZ)

            dx, dy = env.env._pixel2pos(p_x, p_y, graspZ, arena_pos=[0, 0, 0])
            dz = graspZ

            # step 4. move to target point with hover distance
            t_pos = pnt_hem
            t_angle = [result.angle, phi, theta]

            stucked = move_to_6Dpos(env, t_pos, t_angle, cam_offset=[-dx, dy, dz - 0.10], grasp=0.0, level=4.0, render=render)
            if stucked == -1:
                controller_failure += 1
                continue
            print('arrived at the hover point')

            vt_c_im_list.append(vt_c_im)
            vt_d_im_list.append(vt_d_im)
            rot_c_im_list.append(rot_c_im)
            rot_d_im_list.append(rot_d_im)
            vt_g_pos_list.append(t_pos)
            vt_g_euler_list.append(t_angle)
            rot_g_pos_list.append([dx, dy, dz])
            rot_g_euler_list.append([result.angle, 0, 0])

            # step 5. move to real target position
            stucked = approach_to_object(env, t_pos, t_angle, cam_offset=[-dx, dy, dz + 0.01], grasp=0.0, level=2.0, render=render)
            if stucked == -1:
                failure_count += 1
                g_label_list.append(0.0)
                print('Grasping Failed, Success rate: ', success_count / (success_count + failure_count + controller_failure), success_count / (success_count + failure_count), 
                    ', total trial: ', success_count, failure_count, controller_failure)
                stucked = move_to_6Dpos(env, t_pos, t_angle, cam_offset=[-dx, dy, dz - 0.10], grasp=0.0, level=1.0, render=render)
                continue

            # step 6. grasping
            obs = env.env._get_observation()
            print('arrived at:', obs['right_eef_pos'], '(before grasping)')

            stucked = force_gripper(env, grasp=1.0, render=render)
            print("Try grasping!")
            stucked = move_to_6Dpos(env, t_pos, t_angle, cam_offset=[-dx, dy, dz - 0.30], grasp=1.0, level=0.1, render=render)
            print("Go back to the hover point.")

            # new step. collision check - success/failure
            collision = env._check_contact()
            if collision:
                success_count += 1
                g_label_list.append(1.0)
                print('Grasping Success, Success rate: ', success_count / (success_count + failure_count + controller_failure), success_count / (success_count + failure_count), 
                    ', total trial: ', success_count, failure_count, controller_failure)
                move_to_pos(env, release_pos, 0.0, 1.0, level=0.1, render=render)
                force_gripper(env, grasp=0.0, render=render)
            else:
                failure_count += 1
                g_label_list.append(0.0)
                print('Grasping Failed, Success rate: ', success_count/ (success_count + failure_count + controller_failure), success_count / (success_count + failure_count), 
                    ', total trial: ', success_count, failure_count, controller_failure)

        env.env.reset_objects()
        save_file_name = "data/6DoF_grasp/episode_" + str(i) + ".npz"
        np.savez(save_file_name, item_list, pos_list, quat_list, vt_c_im_list, vt_d_im_list, rot_c_im_list, rot_d_im_list, 
            vt_g_pos_list, vt_g_euler_list, rot_g_pos_list, rot_g_euler_list, g_label_list)