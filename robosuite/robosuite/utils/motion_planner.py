import numpy as np
import robosuite.utils.transform_utils as T
from collections.abc import Iterable
from utility import segmentation_green_object, get_camera_image

INIT_ARM_POS = [0.40933302, -1.24377906, 0.68787495, 2.03907987, -0.27229507, 0.8635629,
                0.46484251, 0.12655639, -0.74606415, -0.15337326, 2.04313409, 0.39049096,
                0.30120114, 0.43309788]

def get_target_pos(start_pos, app_angle, delta_pos):
    phi = app_angle[0]
    theta = app_angle[1]
    psi = app_angle[2]

    x1 = np.cos(psi) * np.cos(theta) * np.cos(phi) - np.sin(psi) * np.sin(phi)
    x2 = np.cos(psi) * np.sin(phi) + np.cos(theta) * np.cos(phi) * np.sin(psi)
    x3 = -np.cos(phi) * np.sin(theta)
    y1 = -np.cos(phi) * np.sin(psi) - np.cos(psi) * np.cos(theta) * np.sin(phi)
    y2 = np.cos(psi) * np.cos(phi) - np.cos(theta) * np.sin(psi) * np.sin(phi)
    y3 = np.sin(theta) * np.sin(phi)
    z1 = np.cos(psi) * np.sin(theta)
    z2 = np.sin(psi) * np.sin(theta)
    z3 = np.cos(theta)

    drot = np.array([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]])
    rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
    rotation = rotation.dot(drot)
    target_pos = start_pos + rotation.dot(delta_pos)

    return target_pos

def get_camera_pos(arena_pos, t_angle):
    rotation_ = np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]])

    phi = 0
    theta = t_angle[1] #2
    psi = t_angle[2]   #1

    x1 = np.cos(psi) * np.cos(theta) * np.cos(phi) - np.sin(psi) * np.sin(phi)
    x2 = np.cos(psi) * np.sin(phi) + np.cos(theta) * np.cos(phi) * np.sin(psi)
    x3 = -np.cos(phi) * np.sin(theta)
    y1 = -np.cos(phi) * np.sin(psi) - np.cos(psi) * np.cos(theta) * np.sin(phi)
    y2 = np.cos(psi) * np.cos(phi) - np.cos(theta) * np.sin(psi) * np.sin(phi)
    y3 = np.sin(theta) * np.sin(phi)
    z1 = np.cos(psi) * np.sin(theta)
    z2 = np.sin(psi) * np.sin(theta)
    z3 = np.cos(theta)

    drot = np.array([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]])
    rotation = rotation_.dot(drot)
    mat = np.array([[0, -1, 0],[-1, 0, 0],[0, 0, -1]])
    rotation = rotation.dot(mat)

    #t_pos = arena_pos + np.array([0.0, 0.0, 0.28]) + 0.4 * np.array([np.sin(phi) * np.cos(theta), -np.sin(phi) *  np.sin(theta), np.cos(phi)])
    t_pos = arena_pos + np.array([0, 0, 0.0]) + 0.6 * np.array([np.sin(theta) * np.cos(psi), -np.sin(theta) * np.sin(psi), np.cos(theta)])

    return t_pos, rotation

def get_arm_rotation(t_angle):
    rotation_ = np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]])

    phi = t_angle[0]
    theta = t_angle[1]
    psi = t_angle[2]

    x1 = np.cos(psi) * np.cos(theta) * np.cos(phi) - np.sin(psi) * np.sin(phi)
    x2 = np.cos(psi) * np.sin(phi) + np.cos(theta) * np.cos(phi) * np.sin(psi)
    x3 = -np.cos(phi) * np.sin(theta)
    y1 = -np.cos(phi) * np.sin(psi) - np.cos(psi) * np.cos(theta) * np.sin(phi)
    y2 = np.cos(psi) * np.cos(phi) - np.cos(theta) * np.sin(psi) * np.sin(phi)
    y3 = np.sin(theta) * np.sin(phi)
    z1 = np.cos(psi) * np.sin(theta)
    z2 = np.sin(psi) * np.sin(theta)
    z3 = np.cos(theta)
    
    drot = np.array([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]])
    rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
    rotation = rotation.dot(drot)
    '''drot = np.array([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]])
    rotation = rotation_.dot(drot)
    mat = np.array([[0, -1, 0],[-1, 0, 0],[0, 0, -1]])
    rotation = rotation.dot(mat)'''

    return rotation

def move_to_6Dpos(env, t_pos_left, t_angle_left, t_pos_right, t_angle_right, arm='right', left_grasp=0.0, right_grasp=0.0, level=1.0, render=True):

    if arm == 'left':
        rotation = get_arm_rotation(t_angle_left)   
    elif arm == 'right':
        rotation = get_arm_rotation(t_angle_right)
    elif arm == 'both':
        l_rotation = get_arm_rotation(t_angle_left)
        r_rotation = get_arm_rotation(t_angle_right)
    #rotation_ = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])

    #phi = t_angle[0] #3
    #theta = t_angle[1] #2
    #psi = t_angle[2] #1

    #_, rotation = get_pos_and_rot(_t_pos, phi, theta, psi, cam_offset=cam_offset)

    #rotation = get_arm_rotation(t_angle)
    in_count = 0
    step_count = 0
    action_list = []
    while in_count < 20:

        if arm == 'left':
            pos = env._l_eef_xpos
            current = env._left_hand_orn
            drotation = current.T.dot(rotation)
            dquat = T.mat2quat(drotation)
            dpos = np.array(t_pos_left - pos, dtype=np.float32)
            xyz_action = np.where(abs(dpos) > 0.05, dpos * 1e-2, dpos * 5e-3)
            action = np.concatenate(([0] * 6, [1], xyz_action, dquat * 5e-3, [right_grasp - 1.0], [left_grasp - 1.0]))
            action_list.append(action)

        elif arm == 'right':
            pos = env._r_eef_xpos
            current = env._right_hand_orn
            drotation = current.T.dot(rotation)
            dquat = T.mat2quat(drotation)
            dpos = np.array(t_pos_right - pos, dtype=np.float32)
            xyz_action = np.where(abs(dpos) > 0.05, dpos * 1e-2, dpos * 5e-3)
            action = np.concatenate((xyz_action, dquat * 5e-3, [0] * 6, [1], [right_grasp - 1.0], [left_grasp - 1.0]))
            action_list.append(action)

        elif arm == 'both':
            l_pos = env._l_eef_xpos
            current = env._left_hand_orn
            drotation = current.T.dot(l_rotation)
            dquat = T.mat2quat(drotation)
            l_dpos = np.array(t_pos_left - l_pos, dtype=np.float32)
            xyz_action = np.where(abs(l_dpos) > 0.05, l_dpos * 1e-2, l_dpos * 5e-3)
            action = np.concatenate(([0] * 6, [1], xyz_action, dquat * 5e-3, [right_grasp - 1.0], [left_grasp - 1.0]))

            r_pos = env._r_eef_xpos
            current = env._right_hand_orn
            drotation = current.T.dot(r_rotation)
            dquat = T.mat2quat(drotation)
            r_dpos = np.array(t_pos_right - r_pos, dtype=np.float32)
            xyz_action = np.where(abs(r_dpos) > 0.05, r_dpos * 1e-2, r_dpos * 5e-3)
            action[0:3] = xyz_action
            action[3:7] = dquat * 5e-3
            action_list.append(action)

        obs, reward, done, _ = env.step(action)
        if render:
            env.render()

        '''if arm == 'left':
            next_pos = obs['left_eef_pos']
        elif arm == 'right':
            next_pos = obs['right_eef_pos']'''

        if step_count > 2000:
            '''if arm == 'left':
                print('pos:', obs['left_eef_pos'],', tpos: ', t_pos)
            elif arm == 'right':
                print('pos:', obs['right_eef_pos'],', tpos: ', t_pos)'''
            print("Stucked!")
            #print('Stucked! reset the arms.')
            #env.env.reset_arms(qpos=INIT_ARM_POS)
            #env.env.sim.forward()
            return -1
            
        if (arm == 'left' or arm == 'right') and np.all(abs(dpos) < 0.01 / level): #0.01
            in_count += 1
        elif arm == 'both' and np.all(abs(l_dpos) < 0.01 / level) and np.all(abs(r_dpos) < 0.01 / level):
            in_count += 1
        else:
            in_count = 0
        step_count += 1

    print('move_to_6Dpos success!!')

    return action_list

def move_to_pos(env, t_pos_left, t_pos_right, arm='right', left_grasp=0.0, right_grasp=0.0, level=1.0, render=True):
    rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
    drot = T.rotation_matrix(angle=0.0, direction=[0.0, 0.0, 1.0])[:3, :3]
    rotation = rotation.dot(drot)

    in_count = 0
    step_count = 0
    action_list = []
    while in_count < 5:

        if arm == 'left':
            pos = env._l_eef_xpos
            current = env._left_hand_orn
            drotation = current.T.dot(rotation)
            dquat = T.mat2quat(drotation)
            dpos = np.array(t_pos_left - pos, dtype=np.float32)
            xyz_action = np.where(abs(dpos) > 0.05, dpos * 1e-2, dpos * 5e-3)
            action = np.concatenate(([0] * 6, [1], xyz_action, dquat * 5e-3, [right_grasp - 1.0], [left_grasp - 1.0]))
            action_list.append(action)

        elif arm == 'right':
            pos = env._r_eef_xpos
            current = env._right_hand_orn
            drotation = current.T.dot(rotation)
            dquat = T.mat2quat(drotation)
            dpos = np.array(t_pos_right - pos, dtype=np.float32)
            xyz_action = np.where(abs(dpos) > 0.05, dpos * 1e-2, dpos * 5e-3)
            action = np.concatenate((xyz_action, dquat * 5e-3, [0] * 6, [1], [right_grasp - 1.0], [left_grasp - 1.0]))
            action_list.append(action)

        elif arm == 'both':
            l_pos = env._l_eef_xpos
            current = env._left_hand_orn
            drotation = current.T.dot(rotation)
            dquat = T.mat2quat(drotation)
            l_dpos = np.array(t_pos_left - l_pos, dtype=np.float32)
            xyz_action = np.where(abs(l_dpos) > 0.05, l_dpos * 1e-2, l_dpos * 5e-3)
            action = np.concatenate(([0] * 6, [1], xyz_action, dquat * 5e-3, [right_grasp - 1.0], [left_grasp - 1.0]))

            r_pos = env._r_eef_xpos
            current = env._right_hand_orn
            drotation = current.T.dot(rotation)
            dquat = T.mat2quat(drotation)
            r_dpos = np.array(t_pos_right - r_pos, dtype=np.float32)
            xyz_action = np.where(abs(r_dpos) > 0.05, r_dpos * 1e-2, r_dpos * 5e-3)
            action[0:3] = xyz_action
            action[3:7] = dquat * 5e-3
            action_list.append(action)

        obs, reward, done, _ = env.step(action)
        if render:
            env.render()

        if step_count > 2000:
            print("Stucked!")
            return -1

        if (arm == 'left' or arm == 'right') and np.all(abs(dpos) < 0.01 / level): #0.01
            in_count += 1
        elif arm == 'both' and np.all(abs(l_dpos) < 0.01 / level) and np.all(abs(r_dpos) < 0.01 / level):
            in_count += 1
        else:
            in_count = 0
        step_count += 1

    return action_list

def force_gripper(env, left_grasp=0.0, right_grasp=0.0, render=True):
    action_list = []
    for i in range(0, 200):
        action = np.concatenate(([0] * 6, [1], [0] * 6, [1], [right_grasp - 1.0], [left_grasp - 1.0]))
        action_list.append(action)
        obs, reward, done, _ = env.step(action)
        if render:
            env.render()
    return action_list

def stop_force_gripper(env, t_pos_left, t_angle_left, t_pos_right, t_angle_right, arm='right', left_grasp=0.0, right_grasp=0.0, level=1.0, render=True):
    action_list = []
    if arm == 'left':
        rotation = get_arm_rotation(t_angle_left)   
    elif arm == 'right':
        rotation = get_arm_rotation(t_angle_right)
    elif arm == 'both':
        l_rotation = get_arm_rotation(t_angle_left)
        r_rotation = get_arm_rotation(t_angle_right)

    for i in range(0, 200):
        if arm == 'left':
            pos = env._l_eef_xpos
            current = env._left_hand_orn
            drotation = current.T.dot(rotation)
            dquat = T.mat2quat(drotation)
            dpos = np.array(t_pos_left - pos, dtype=np.float32)
            xyz_action = np.where(abs(dpos) > 0.05, dpos * 1e-2, dpos * 5e-3)
            action = np.concatenate(([0] * 6, [1], xyz_action, dquat * 5e-3, [right_grasp - 1.0], [left_grasp - 1.0]))
            action_list.append(action)

        elif arm == 'right':
            pos = env._r_eef_xpos
            current = env._right_hand_orn
            drotation = current.T.dot(rotation)
            dquat = T.mat2quat(drotation)
            dpos = np.array(t_pos_right - pos, dtype=np.float32)
            xyz_action = np.where(abs(dpos) > 0.05, dpos * 1e-2, dpos * 5e-3)
            action = np.concatenate((xyz_action, dquat * 5e-3, [0] * 6, [1], [right_grasp - 1.0], [left_grasp - 1.0]))
            action_list.append(action)

        elif arm == 'both':
            l_pos = env._l_eef_xpos
            current = env._left_hand_orn
            drotation = current.T.dot(l_rotation)
            dquat = T.mat2quat(drotation)
            l_dpos = np.array(t_pos_left - l_pos, dtype=np.float32)
            xyz_action = np.where(abs(l_dpos) > 0.05, l_dpos * 1e-2, l_dpos * 5e-3)
            action = np.concatenate(([0] * 6, [1], xyz_action, dquat * 5e-3, [right_grasp - 1.0], [left_grasp - 1.0]))

            r_pos = env._r_eef_xpos
            current = env._right_hand_orn
            drotation = current.T.dot(r_rotation)
            dquat = T.mat2quat(drotation)
            r_dpos = np.array(t_pos_right - r_pos, dtype=np.float32)
            xyz_action = np.where(abs(r_dpos) > 0.05, r_dpos * 1e-2, r_dpos * 5e-3)
            action[0:3] = xyz_action
            action[3:7] = dquat * 5e-3
            action_list.append(action)

        obs, reward, done, _ = env.step(action)
        if render:
            env.render()
    return action_list

def object_pass(env, init_obj_pos, float_pos, init_left_arm_pos=np.array([0.4, 0.6, 1.0]), init_right_arm_pos=np.array([0.4, -0.6, 1.0]), render=True):
    left_t_pos, right_t_pos = init_left_arm_pos, init_right_arm_pos
    stucked = move_to_pos(env, left_t_pos, right_t_pos, arm='both', level=1.0, render=render)
    if stucked == -1:
        return -1

    right_arm_camera_id = env.sim.model.camera_name2id("eye_on_right_wrist")
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
    #sel_start = time.time()
    
    print("Try FC-GQ-CNN")
    [result, depths, d_im], rot_c_im, rot_d_im = env.env.gqcnn(arm='right', vis_on=False, num_candidates=10) ## vis_on 

    if isinstance(result, Iterable):
        sample_grasp_idx = np.random.randint(10, size=1)
        result = result[sample_grasp_idx[0]].grasp
    else:
        if result is None:
            return -1
        else:
            result = result.grasp

    p_x, p_y = result.center
    graspZ = result.depth

    dx, dy = env.env._pixel2pos(p_x, p_y, graspZ, arena_pos=[0, 0, 0])
    dz = graspZ

    right_t_angle = [result.angle, phi, theta]
    right_t_pos = get_target_pos(pnt_hem, [0, phi, theta], [-dx, dy, dz - 0.1])
    stucked = move_to_6Dpos(env, None, None, right_t_pos, right_t_angle, arm='right', level=4.0, render=render)
    if stucked == -1:
        return -1

    right_t_pos = get_target_pos(pnt_hem, [0, phi, theta], [-dx, dy, dz + 0.01])
    stucked = move_to_6Dpos(env, None, None, right_t_pos, right_t_angle, arm='right', level=4.0, render=render)
    if stucked == -1:
        return -1

    stucked = force_gripper(env, right_grasp=1.0, render=render)
    right_t_pos = get_target_pos(pnt_hem, [0, phi, theta], [-dx, dy, dz - 0.3])
    stucked = move_to_6Dpos(env, None, None, right_t_pos, right_t_angle, arm='right', right_grasp=1.0, level=0.1, render=render)

    collision = env._check_contact(arm='right')
    if collision:
        right_t_pos = float_pos
        right_t_angle = np.array([2 * np.pi * np.random.random(), np.pi / 3.0, np.pi * np.random.random()])
        stucked = move_to_6Dpos(env, None, None, right_t_pos, right_t_angle, arm='right', right_grasp=1.0, level=1.0, render=render)
        if stucked == -1:
            return -1
    else:
        #move_to_pos(env, None, np.array([0.4, -0.6, 1.0]), arm='right', level=1.0, render=render)
        return -1

    camera_obs = env.sim.render(
        camera_name="eye_on_left_wrist",
        width=env.camera_width,
        height=env.camera_height,
        depth=env.camera_depth
    )

    phi = np.pi / 10.0 * np.random.random()
    theta = np.pi * np.random.random() + np.pi
    camera_pos, camera_rot_mat = get_camera_pos(right_t_pos, np.array([0, phi, theta]))

    # CEM best viewpoint
    #sel_start = time.time()
    
    print("Try FC-GQ-CNN")
    rot_c_im, rot_d_im = get_camera_image(env, camera_pos, camera_rot_mat, arm='left', vis_on=False)
    mask_rot_d_im = segmentation_green_object(rot_c_im, rot_d_im, clip=True)
    result, depths, d_im = env.policy.evaluate_gqcnn(rot_c_im, mask_rot_d_im, vis_on=True, num_candidates=10)

    if isinstance(result, Iterable):
        sample_grasp_idx = np.random.randint(10, size=1)
        result = result[sample_grasp_idx[0]].grasp
    else:
        if result is None:
            return -1
        else:
            result = result.grasp

    p_x, p_y = result.center
    graspZ = result.depth

    dx, dy = env.env._pixel2pos(p_x, p_y, graspZ, arena_pos=[0, 0, 0])
    dz = graspZ

    left_t_angle = [result.angle, phi, theta]

    left_t_pos = get_target_pos(pnt_hem, [0, phi, theta], [-dx, dy, dz - 0.3])
    stucked = move_to_6Dpos(env, left_t_pos, left_t_angle, right_t_pos, right_t_angle, arm='both', right_grasp=1.0, level=4.0, render=render)
    if stucked == -1:
        return -1
    left_t_pos = get_target_pos(pnt_hem, [0, phi, theta], [-dx, dy, dz - 0.1])
    stucked = move_to_6Dpos(env, left_t_pos, left_t_angle, right_t_pos, right_t_angle, arm='both', right_grasp=1.0, level=4.0, render=render)
    if stucked == -1:
        return -1

    left_t_pos = get_target_pos(pnt_hem, [0, phi, theta], [-dx, dy, dz + 0.01])
    stucked = move_to_6Dpos(env, left_t_pos, left_t_angle, right_t_pos, right_t_angle, arm='both', right_grasp=1.0, level=4.0, render=render)
    if stucked == -1:
        return -1

    force_gripper(env, left_grasp=1.0, right_grasp=1.0, render=render)
    force_gripper(env, left_grasp=1.0, right_grasp=0.0, render=render)

    right_t_pos = get_target_pos(right_t_pos, right_t_angle, [0.0, 0.0, -0.3])
    stucked = move_to_6Dpos(env, None, None, right_t_pos, right_t_angle, arm='right', left_grasp=1.0, level=0.1, render=render)
    if stucked == -1:
        return -1

    left_t_pos = get_target_pos(pnt_hem, [0, phi, theta], [-dx, dy, dz - 0.3])
    stucked = move_to_6Dpos(env, left_t_pos, left_t_angle, right_t_pos, right_t_angle, arm='both', left_grasp=1.0, level=0.1, render=render)
    if stucked == -1:
        return -1
    stucked = move_to_pos(env, np.array([0.4, 0.6, 1.0]), np.array([0.4, -0.6, 1.0]), arm='both', left_grasp=1.0, level=1.0, render=render)
    if stucked == -1:
        return -1

    collision = env._check_contact(arm='left')
    if collision:
        force_gripper(env, left_grasp=0.0, render=render)
    else:
        return -1