import argparse
import numpy as np
import time
from collections.abc import Iterable
import robosuite
import robosuite.utils.transform_utils as T
from robosuite.wrappers import IKWrapper
import matplotlib.pyplot as plt
from robosuite.utils.mjcf_utils import array_to_string, string_to_array
import json
import os

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

    files_dir = "/home/rllab/robosuite-gqcnn/robosuite/robosuite/scripts/data/iros2020"
    files = os.listdir(files_dir)
    files.sort()
    for f in files:
        data = np.load(os.path.join(files_dir, f))
        vt_c_im = data["arr_0"]
        vt_d_im = data["arr_1"]
        mask_vt_d_im = data["arr_2"]
        object_mask = data["arr_3"]
        arm_mask = data["arr_4"]
        rot_c_im = data["arr_5"]
        rot_d_im = data["arr_6"]
        mask_rot_d_im = data["arr_7"]
        gqcnn_t_pos = data["arr_8"]
        gqcnn_t_angle = data["arr_9"]
        label = data["arr_10"]
        stuck = data["arr_11"]

        delta_pos = gqcnn_t_pos - [0.6, 0.0, 0.77]
        res = 2 * 0.6 * np.tan(np.pi / 8.0) / 256.0
        delta_y, delta_x, _ = delta_pos / res
        pos_y = delta_y + 128
        pos_x = 128 - delta_x

        '''crop_c_image = vt_c_im[(int)(256 - pos_y) - 48:(int)(256 - pos_y) + 48, (int)(pos_x) - 48:(int)(pos_x) + 48] 
        crop_d_image = vt_d_im[(int)(256 - pos_y) - 48:(int)(256 - pos_y) + 48, (int)(pos_x) - 48:(int)(pos_x) + 48]

        if crop_c_image.shape[0] != 96 or crop_c_image.shape[1] != 96:
            continue
        else:
            result, depths, d_im = env.env.policy.evaluate_gqcnn(crop_c_image, crop_d_image, vis_on=True, num_candidates=2)'''

        crop_c_image = vt_c_im[(int)(256 - pos_y) - 58:(int)(256 - pos_y) + 58, (int)(pos_x) - 58:(int)(pos_x) + 58] 
        crop_d_image = vt_d_im[(int)(256 - pos_y) - 58:(int)(256 - pos_y) + 58, (int)(pos_x) - 58:(int)(pos_x) + 58]
        max_q_value = -1

        if crop_c_image.shape[0] != 116 or crop_c_image.shape[1] != 116:
            continue
        else:
            result, depths, d_im = env.env.policy.evaluate_gqcnn(crop_c_image, crop_d_image, vis_on=False, num_candidates=3)
            q_value_list = [r.q_value for r in result]

            p_x, p_y = result[np.argmax(q_value_list)].grasp.center
            angle = result[np.argmax(q_value_list)].grasp.angle
            p_x = p_x - 58
            p_y = p_y - 58

            pos_y = pos_y - p_y
            pos_x = pos_x + p_x

            crop_c_image = vt_c_im[(int)(256 - pos_y) - 48:(int)(256 - pos_y) + 48, (int)(pos_x) - 48:(int)(pos_x) + 48] 
            crop_d_image = vt_d_im[(int)(256 - pos_y) - 48:(int)(256 - pos_y) + 48, (int)(pos_x) - 48:(int)(pos_x) + 48]

            delta_pos_x = res * (pos_y - 128)
            delta_pos_y = res * (128 - pos_x)

            gqcnn_t_pos = np.asarray([0.6, 0.0, 0.77]) + np.asarray([delta_pos_x, delta_pos_y, 0.0])
            print(gqcnn_t_pos)
            gqcnn_t_angle = [angle, 0.0, 0.0]

            np.savez("data/iros2020_gqcnn_ring/" + f, vt_c_im, vt_d_im, mask_vt_d_im, object_mask, arm_mask, rot_c_im, rot_d_im, mask_rot_d_im, gqcnn_t_pos, gqcnn_t_angle, label, 0.0)

            """crop_c_image = vt_c_im[(int)(256 - pos_y + p_y) - 48:(int)(256 - pos_y + p_y) + 48, (int)(pos_x + p_x) - 48:(int)(pos_x + p_x) + 48] 
            crop_d_image = vt_d_im[(int)(256 - pos_y + p_y) - 48:(int)(256 - pos_y + p_y) + 48, (int)(pos_x + p_x) - 48:(int)(pos_x + p_x) + 48]"""

            '''for i in range(3):
                p_x, p_y = result[i].grasp.center
                q_value = result[i].q_value'''


        #np.savez("data/iros2020/episode_" + str(save_num) + ".npz", vt_c_im, vt_d_im, mask_vt_d_im, object_mask, arm_mask, rot_c_im, rot_d_im, mask_rot_d_im, gqcnn_t_pos, gqcnn_t_angle, label, 0.0)

    '''[result, depths, d_im], rot_c_im, rot_d_im = env.env.gqcnn(arm='right', vis_on=False, num_candidates=20) ## vis_on 

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

            right_t_angle = [result.angle, phi, theta]'''