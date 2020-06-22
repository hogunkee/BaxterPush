import numpy as np
import robosuite.utils.transform_utils as T

INIT_ARM_POS = [0.40933302, -1.24377906, 0.68787495, 2.03907987, -0.27229507, 0.8635629,
                0.46484251, 0.12655639, -0.74606415, -0.15337326, 2.04313409, 0.39049096,
                0.30120114, 0.43309788]

def get_pos_and_rot(pos, phi, theta, psi, cam_offset=np.array([0, 0, 0])):
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
    rotation_ = np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]])
    rotation = rotation_.dot(drot)
    t_pos = pos + rotation.dot(cam_offset)
    return t_pos, rotation

class PID:
    def __init__(self, current_time, P=1e-3, I=0.0, D=0.0):
        self.Kp = P
        self.Ki = I
        self.Kd = D
        self.current_time = current_time
        self.last_time = self.current_time
        self.clear()

    def clear(self):
        self.PTerm = 0.0
        self.ITerm = 0.0
        self.DTerm = 0.0
        self.last_error = None
        self.int_error = np.zeros((7))

    def update(self, error, delta_time):
        if self.last_error is None:
            self.last_error = error
        delta_error = error - self.last_error
        #self.current_time = current_time
        #delta_time = self.current_time - self.last_time

        self.PTerm = self.Kp * error
        self.ITerm += error * delta_time
        self.DTerm = delta_error / delta_time
        #self.last_time = self.current_time
        self.last_error = error

        output = self.PTerm + self.Ki * self.ITerm + self.Kd * self.DTerm
        return output

    '''def update(self, feedback_value, current_time):
        error = self.SetPoint - feedback_value
        delta_error = error - self.last_error
        self.current_time = current_time
        delta_time = self.current_time - self.last_time

        self.PTerm = self.Kp * error
        self.ITerm += error * delta_time
        self.DTerm = delta_error / delta_time
        self.last_time = self.current_time
        self.last_error = error

        self.output = self.PTerm + self.Ki * self.ITerm + self.Kd * self.DTerm'''

def move_to_6Dpos_2(env, _t_pos, t_angle, cam_offset=np.array([0, 0, 0]), grasp=0.0, level=1.0, render=True):
    rotation_ = np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]])

    phi = 0  #3
    theta = t_angle[1] #2
    psi = t_angle[2] #1

    t_pos, rotation = get_pos_and_rot(_t_pos, phi, theta, psi, cam_offset=cam_offset)

    ground_depth = env.bin_pos[2]
    if t_pos[2] < ground_depth:
        print("t_pos: ", t_pos)
        print("Cannot grasp on the ground. pass")
        return -1
        # clipping

    phi = t_angle[0]
    _, rotation = get_pos_and_rot(_t_pos, phi, theta, psi, cam_offset=cam_offset)
    pid_controller = PID(0.0, P=5.0e-3, I=0.0, D=0.0)

    in_count = 0
    step_count = 0
    action_list = []
    while in_count < 20:
        pos = env._r_eef_xpos

        current = env._right_hand_orn
        drotation = current.T.dot(rotation)
        dquat = T.mat2quat(drotation)

        dpos = np.array(t_pos - pos, dtype=np.float32)
        xyz_action = np.where(abs(dpos) > 0.05, dpos * 1e-2, dpos * 5e-3)
        action = np.concatenate((xyz_action, dquat * 5e-3, [0] * 6, [1], [grasp - 1.0], [-1.0]))
        #action = pid_controller.update(np.concatenate((dpos, dquat)), env.control_timestep)
        #action = np.concatenate((action, [0] * 6, [1], [grasp - 1.0], [-1.0]))
        action_list.append(action)

        obs, reward, done, _ = env.step(action)
        if render:
            env.render()
            
        if step_count > 1000:
            print('pos:', obs['right_eef_pos'],', tpos: ', t_pos)
            print('Stucked! reset the arms.')
            return -1

        if np.all(abs(dpos) < 0.01 / level): #0.01
            in_count += 1
        else:
            in_count = 0
        step_count += 1

    print('move_to_6Dpos success!!')

    return action_list

def approach_to_object(env, _t_pos, t_angle, cam_offset=np.array([0, 0, 0]), grasp=0.0, level=1.0, render=True):
    rotation_ = np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]])
    phi = 0  #3
    theta = t_angle[1] #2
    psi = t_angle[2] #1

    t_pos, _ = get_pos_and_rot(_t_pos, phi, theta, psi, cam_offset=cam_offset)

    '''ground_depth = env.bin_pos[2]
    if t_pos[2] < ground_depth:
        print("t_pos: ", t_pos)
        print("Cannot grasp on the ground. pass")
        return -1
    t_pos[2] = np.clip(t_pos[2], ground_depth, None)'''
        # clipping

    phi = t_angle[0]
    _, rotation = get_pos_and_rot(_t_pos, phi, theta, psi, cam_offset=cam_offset)

    in_count = 0
    step_count = 0
    stop_count = 0
    action_list = []
    pid_controller = PID(0.0, P=5.0e-3, I=5.0e-6, D=5.0e-6)
    #pid_controller = PID(0.0, P=5.0e-3, I=0.0, D=0.0)
    while in_count < 10:
        pos = env._r_eef_xpos
        prev_pos = np.copy(env._r_eef_xpos)

        current = env._right_hand_orn
        drotation = current.T.dot(rotation)
        dquat = T.mat2quat(drotation)

        dpos = np.array(t_pos - pos, dtype=np.float32)
        action = pid_controller.update(np.concatenate((dpos, dquat)), env.control_timestep)
        action = np.concatenate((action, [0] * 6, [1], [grasp - 1.0], [-1.0]))
        action_list.append(action)

        obs, reward, done, _ = env.step(action)
        next_pos = obs['right_eef_pos']
        if render:
            env.render()
            
        if step_count > 2000:
            print('pos:', obs['right_eef_pos'],', tpos: ', t_pos)
            print('Stucked!')
            return -1

        if stop_count > 200:
            print('pos:', obs['right_eef_pos'],', tpos: ', t_pos)
            print("Stucked!")
            return -1

        if np.all(abs(dpos) < 0.01 / level): #0.01
            in_count += 1
        else:
            in_count = 0

        #print(np.linalg.norm(next_pos - prev_pos), next_pos, prev_pos)
        if np.linalg.norm(next_pos - prev_pos) < 1.0e-5:
            stop_count += 1
        else:
            stop_count = 0
        step_count += 1

    print('move_to_6Dpos success!!')

    return action_list

def move_to_6Dpos(env, _t_pos, t_angle, cam_offset=np.array([0, 0, 0]), grasp=0.0, level=1.0, render=True):
    rotation_ = np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]])

    phi = 0  #3
    theta = t_angle[1] #2
    psi = t_angle[2] #1

    t_pos, _ = get_pos_and_rot(_t_pos, phi, theta, psi, cam_offset=cam_offset)

    '''ground_depth = env.bin_pos[2]
    if t_pos[2] < ground_depth:
        print("t_pos: ", t_pos)
        print("Cannot grasp on the ground. pass")
        return -1
    t_pos[2] = np.clip(t_pos[2], ground_depth, None)'''
    # clipping

    phi = t_angle[0]
    _, rotation = get_pos_and_rot(_t_pos, phi, theta, psi, cam_offset=cam_offset)

    in_count = 0
    step_count = 0
    action_list = []
    pid_controller = PID(0.0, P=5.0e-3, I=5.0e-6, D=5.0e-6)
    #pid_controller = PID(0.0, P=5.0e-3, I=0.0, D=0.0)
    while in_count < 20:
        pos = env._r_eef_xpos
        current = env._right_hand_orn
        drotation = current.T.dot(rotation)
        dquat = T.mat2quat(drotation)

        dpos = np.array(t_pos - pos, dtype=np.float32)

        #xyz_action = np.where(abs(dpos) > 0.05, dpos * 1e-2, dpos * 5e-3)
        #action = np.concatenate((xyz_action, dquat * 5e-3, [0] * 6, [1], [grasp - 1.0], [-1.0]))
        action = pid_controller.update(np.concatenate((dpos, dquat)), env.control_timestep)
        action = np.concatenate((action, [0] * 6, [1], [grasp - 1.0], [-1.0]))
        #xyz_action = np.where(abs(dpos) > 0.05, dpos * 1e-2, dpos * 1e-2)
        #action = np.concatenate((xyz_action, dquat * 1e-2, [0] * 6, [1], [grasp - 1.0], [-1.0]))
        action_list.append(action)
        prev_pos = np.copy(env._r_eef_xpos)
        obs, reward, done, _ = env.step(action)
        if render:
            env.render()
        next_pos = obs['right_eef_pos']

        if step_count > 2000:
            print('pos:', obs['right_eef_pos'],', tpos: ', t_pos)
            print("Stucked!")
            #print('Stucked! reset the arms.')
            #env.env.reset_arms(qpos=INIT_ARM_POS)
            #env.env.sim.forward()
            return -1

        if np.all(abs(dpos) < 0.01 / level): #0.01
            in_count += 1
        else:
            in_count = 0
        step_count += 1

    print('move_to_6Dpos success!!')

    return action_list

def move_to_6Dpos_larm(env, _t_pos, t_angle, cam_offset=np.array([0, 0, 0]), grasp=0.0, level=1.0, render=True):
    rotation_ = np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]])

    phi = t_angle[0]  #3
    theta = t_angle[1] #2
    psi = t_angle[2] #1

    # get robot arm target pose and rotation
    t_pos, _ = get_pos_and_rot(_t_pos, 0.0, theta, psi, cam_offset=cam_offset)
    _, rotation = get_pos_and_rot(_t_pos, phi, theta, psi, cam_offset=cam_offset)

    '''ground_depth = env.bin_pos[2]
    if t_pos[2] < ground_depth:
        print("t_pos: ", t_pos)
        print("Cannot grasp on the ground. pass")
        return -1
    t_pos[2] = np.clip(t_pos[2], ground_depth, None)'''

    in_count = 0
    step_count = 0
    action_list = []

    '''env.env.reset_arms(qpos=INIT_ARM_POS)
    env.env.sim.forward()

    rotation = np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]])
    drot = T.rotation_matrix(angle=t_angle, direction=[0., 0., 1.])[:3, :3]
    rotation = rotation.dot(drot)'''

    while in_count < 1:
        pos = env._l_eef_xpos # obs['left_eef_pos']

        current = env._left_hand_orn
        drotation = current.T.dot(rotation)
        dquat = T.mat2quat(drotation)

        dpos = np.array(t_pos - pos, dtype=np.float32)
        xyz_action = np.where(abs(dpos) > 0.05, dpos * 1e-2, dpos * 5e-3)

        #action = np.concatenate(([0] * 7, xyz_action, dquat * 5e-3, [grasp - 1.0], [-1.0]))
        action = np.concatenate(([0] * 6, [1], xyz_action, dquat * 5e-3, [grasp - 1.0], [-1.0]))
        action_list.append(action)

        obs, reward, done, _ = env.step(action)
        if render:
            env.render()
            
        if step_count > 2000:
            print('pos:', obs['left_eef_pos'])
            print('Stucked! reset the arms.')
            #env.env.sim.data.qpos[env._ref_joint_pos_indexes] = qpos
            #env.env.reset_arms(qpos=INIT_ARM_POS)
            #env.env.sim.forward()
            return -1

        if np.all(abs(dpos) < 0.01 / level) and np.all(abs(dquat - np.array([0, 0, 0, 1])) < 0.01):
            in_count += 1
        else:
            in_count = 0
        step_count += 1

    return action_list

"""def move_to_6Dpos(env, _t_pos, t_angle, cam_offset=np.array([0, 0, 0]), grasp=0.0, level=1.0, render=True):
    rotation_ = np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]])

    phi = 0  #3
    theta = t_angle[1] #2
    psi = t_angle[2] #1

    # Euler angles: ZY'Z''
    x1 = np.cos(psi) * np.cos(theta) * np.cos(phi) - np.sin(psi) * np.sin(phi)
    x2 = np.cos(psi) * np.sin(phi) + np.cos(theta) * np.cos(phi) * np.sin(psi)
    x3 = -np.cos(phi) * np.sin(theta)
    y1 = -np.cos(phi) * np.sin(psi) - np.cos(psi) * np.cos(theta) * np.sin(phi)
    y2 = np.cos(psi) * np.cos(phi) - np.cos(theta) * np.sin(psi) * np.sin(phi)
    y3 = np.sin(theta) * np.sin(phi)
    z1 = np.cos(psi) * np.sin(theta)
    z2 = np.sin(psi) * np.sin(theta)
    z3 = np.cos(theta)

    # drot = T.rotation_matrix(angle=t_angle, direction=[0., 0., 1.])[:3, :3]
    drot = np.array([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]])
    rotation = rotation_.dot(drot)
    t_pos = _t_pos + rotation.dot(cam_offset)

    ground_depth = env.bin_pos[2]
    if t_pos[2] < ground_depth:
        print("t_pos: ", t_pos)
        print("Cannot grasp on the ground. pass")
        return -1
    t_pos[2] = np.clip(t_pos[2], ground_depth, None)
        # clipping

    phi = t_angle[0] 
    # Euler angles: ZY'Z''
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

    in_count = 0
    step_count = 0
    action_list = []
    pid_controller = PID(0.0, P=5.0e-3, I=5.0e-6, D=5.0e-6)
    #pid_controller = PID(0.0, P=5.0e-3, I=0.0, D=0.0)
    while in_count < 20:
        pos = env._r_eef_xpos
        current = env._right_hand_orn
        drotation = current.T.dot(rotation)
        dquat = T.mat2quat(drotation)

        dpos = np.array(t_pos - pos, dtype=np.float32)

        #xyz_action = np.where(abs(dpos) > 0.05, dpos * 1e-2, dpos * 5e-3)
        #action = np.concatenate((xyz_action, dquat * 5e-3, [0] * 6, [1], [grasp - 1.0], [-1.0]))
        action = pid_controller.update(np.concatenate((dpos, dquat)), env.control_timestep)
        action = np.concatenate((action, [0] * 6, [1], [grasp - 1.0], [-1.0]))
        #xyz_action = np.where(abs(dpos) > 0.05, dpos * 1e-2, dpos * 1e-2)
        #action = np.concatenate((xyz_action, dquat * 1e-2, [0] * 6, [1], [grasp - 1.0], [-1.0]))
        action_list.append(action)
        prev_pos = np.copy(env._r_eef_xpos)
        obs, reward, done, _ = env.step(action)
        if render:
            env.render()
        next_pos = obs['right_eef_pos']

        if step_count > 2000:
            print('pos:', obs['right_eef_pos'],', tpos: ', t_pos)
            print("Stucked!")
            #print('Stucked! reset the arms.')
            #env.env.reset_arms(qpos=INIT_ARM_POS)
            #env.env.sim.forward()
            return -1

        if np.all(abs(dpos) < 0.01 / level): #0.01
            in_count += 1
        else:
            in_count = 0

    print('move_to_6Dpos success!!')

    return action_list"""

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

def apply_actions(env, action_list, render=True):
    for a in action_list:
        obs, reward, done, _ = env.step(a)
        if render:
            env.render()
    print('arrived at:', obs['right_eef_pos'])
    return

def move_to_pos(env, t_pos, t_angle=0.0, grasp=0.0, level=1.0, render=True):
    rotation = np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]])
    drot = T.rotation_matrix(angle=t_angle, direction=[0., 0., 1.])[:3, :3]
    rotation = rotation.dot(drot)

    in_count = 0
    step_count = 0
    action_list = []
    while in_count < 1:
        pos = env._r_eef_xpos # obs['right_eef_pos']

        current = env._right_hand_orn
        drotation = current.T.dot(rotation)
        dquat = T.mat2quat(drotation)

        dpos = np.array(t_pos - pos, dtype=np.float32)
        xyz_action = np.where(abs(dpos) > 0.05, dpos * 1e-2, dpos * 5e-3)

        action = np.concatenate((xyz_action, dquat * 5e-3, [0] * 6, [1], [grasp - 1.0], [-1.0]))
        action_list.append(action)

        obs, reward, done, _ = env.step(action)
        if render:
            env.render()
            
        if step_count > 2000:
            print('pos:', obs['right_eef_pos'])
            print('Stucked! reset the arms.')
            env.env.reset_arms(qpos=INIT_ARM_POS)
            env.env.sim.forward()
            return -1

        if np.all(abs(dpos) < 0.01 / level) and np.all(abs(dquat - np.array([0, 0, 0, 1])) < 0.01 / level):
            in_count += 1
        else:
            in_count = 0
        step_count += 1

    return action_list

def move_to_pos_larm(env, t_pos, t_angle=0.0, grasp=0.0, level=1.0, render=True):
    env.env.reset_arms(qpos=INIT_ARM_POS)
    env.env.sim.forward()

    rotation = np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]])
    drot = T.rotation_matrix(angle=t_angle, direction=[0., 0., 1.])[:3, :3]
    rotation = rotation.dot(drot)

    in_count = 0
    step_count = 0
    action_list = []
    while in_count < 1:
        pos = env._l_eef_xpos # obs['left_eef_pos']

        current = env._left_hand_orn
        drotation = current.T.dot(rotation)
        dquat = T.mat2quat(drotation)

        dpos = np.array(t_pos - pos, dtype=np.float32)
        xyz_action = np.where(abs(dpos) > 0.05, dpos * 1e-2, dpos * 5e-3)

        #action = np.concatenate(([0] * 7, xyz_action, dquat * 5e-3, [grasp - 1.0], [-1.0]))
        action = np.concatenate(([0] * 6, [1], xyz_action, dquat * 5e-3, [grasp - 1.0], [-1.0]))
        action_list.append(action)

        obs, reward, done, _ = env.step(action)
        if render:
            env.render()
            
        if step_count > 2000:
            print('pos:', obs['left_eef_pos'])
            print('Stucked! reset the arms.')
            #env.env.sim.data.qpos[env._ref_joint_pos_indexes] = qpos
            #env.env.reset_arms(qpos=INIT_ARM_POS)
            #env.env.sim.forward()
            return -1

        if np.all(abs(dpos) < 0.01 / level) and np.all(abs(dquat - np.array([0, 0, 0, 1])) < 0.01):
            in_count += 1
        else:
            in_count = 0
        step_count += 1

    return action_list

def move_to_pos_lrarm(env, lt_pos, rt_pos, t_angle=0.0, grasp=0.0, level=1.0, render=True):
    rotation = np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]])
    drot = T.rotation_matrix(angle=t_angle, direction=[0., 0., 1.])[:3, :3]
    rotation = rotation.dot(drot)

    in_count = 0
    step_count = 0
    action_list = []
    while in_count < 5:
        l_pos = env._l_eef_xpos # obs['left_eef_pos']
        current = env._left_hand_orn
        drotation = current.T.dot(rotation)
        dquat = T.mat2quat(drotation)

        l_dpos = np.array(lt_pos - l_pos, dtype=np.float32)
        xyz_action = np.where(abs(l_dpos) > 0.05, l_dpos * 1e-2, l_dpos * 5e-3)

        #action = np.concatenate(([0] * 7, xyz_action, dquat * 5e-3, [grasp - 1.0], [-1.0]))
        action = np.concatenate(([0] * 6, [1], xyz_action, dquat * 5e-3, [grasp - 1.0], [-1.0]))

        r_pos = env._r_eef_xpos
        current = env._right_hand_orn
        drotation = current.T.dot(rotation)
        dquat = T.mat2quat(drotation)

        r_dpos = np.array(rt_pos - r_pos, dtype=np.float32)
        xyz_action = np.where(abs(r_dpos) > 0.05, r_dpos * 1e-2, r_dpos * 5e-3)

        action[0:3] = xyz_action
        action[3:7] = dquat * 5e-3
        action_list.append(action)

        obs, reward, done, _ = env.step(action)
        if render:
            env.render()
            
        if step_count > 2000:
            print('pos:', obs['right_eef_pos'])
            print('Stucked! reset the arms.')
            env.env.reset_arms(qpos=INIT_ARM_POS)
            env.env.sim.forward()
            return -1

        if np.all(abs(l_dpos) < 0.01 / level) and np.all(abs(r_dpos) < 0.01 / level):
            in_count += 1
        else:
            in_count = 0
        step_count += 1

    return action_list

def force_gripper(env, grasp=0.0, render=True):
    action_list = []
    for i in range(0, 200):
        action = np.concatenate(([0] * 6, [1], [0] * 6, [1], [grasp - 1.0], [-1.0]))
        action_list.append(action)
        obs, reward, done, _ = env.step(action)
        if render:
            env.render()
    return action_list