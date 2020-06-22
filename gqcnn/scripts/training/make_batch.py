import numpy as np
import os
import pandas as pd

version = '2.1'# 'FC'
input_dir = '/home/gun/Desktop/robosuite/robosuite/robosuite/scripts/data/GQ2.1'
target_dir = '/home/gun/Desktop/newdata'
inner_dir = [input_dir]
#inner_dir = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
#inner_dir = [os.path.join(input_dir, d) for d in inner_dir if version in d and 'GQ' in d and not 'old' in d]

bs = 1000 #FC: 100
target_name = 'GQ2.1_finetune'
target_path = os.path.join(target_dir, target_name)
if target_name not in os.listdir(target_dir):
    os.mkdir(target_path)


num_success = 0
num_failure = 0
df_list = []

if version!='FC':
    date_index = 2
    crop_size = 32
else:
    date_index = 3
    crop_size = 96

for _dir in inner_dir:
    sdf = pd.read_csv(os.path.join(_dir, 'success_depth.csv'), delimiter='\t', header=None)
    fdf = pd.read_csv(os.path.join(_dir, 'failure_depth.csv'), delimiter='\t', header=None)
    num_success += len(sdf)
    num_failure += len(fdf)
    
    sdf[0] = [os.path.join(_dir, 'success_' + str(s) + '.npy') for s in list(sdf[0])]
    fdf[0] = [os.path.join(_dir, 'failure_' + str(s) + '.npy') for s in list(fdf[0])]
    sdf['metric'] = 1.0
    fdf['metric'] = 0.0
    
    df_list.append(sdf)
    df_list.append(fdf)

df = pd.concat(df_list)
df = df.sort_values(by=date_index, ascending=True).reset_index(drop=True)


## depth image ##
tf_depth_ims = None
for npy_path in df[0]:
    depth_im = np.load(npy_path)
    depth_im = depth_im.reshape([crop_size, crop_size, 1])
    if tf_depth_ims is None:
        tf_depth_ims = np.array([depth_im])
    else:
        tf_depth_ims = np.concatenate([tf_depth_ims, [depth_im]])
        
## gripper depth: Z ##
if version=='2.0':
    hand_poses = np.zeros([len(df), 7])
else:
    hand_poses = np.zeros([len(df), 6])
hand_poses[:, 2] = np.array(df[1])

## grasp angle ##
if version=='FC':
    hand_poses[:, 3] = np.array(df[2])

## grasp quality ##    
metric = np.array(df['metric'])

if version=='2.0':
    fname = ['depth_ims_tf_table_', 'hand_poses_', 'robust_ferrari_canny_']
elif version=='2.1':
    fname = ['depth_ims_tf_table_', 'hand_poses_', 'grasp_metrics_']
elif version=='FC':
    fname = ['tf_depth_ims_', 'grasps_', 'grasp_metrics_']

print('[ Model %s ]'%version)
print('success:', num_success)
print('failure:', num_failure)
print('num_data:', num_success + num_failure)
print('success rate: %.3f' %(100*num_success/(num_success+num_failure)))


## save batches into npz format ##
for i in range((len(df)+bs-1)//bs):
    with open(os.path.join(target_path, fname[0]+'%05d.npz'%i), 'wb') as f_dim:
        np.savez(f_dim, tf_depth_ims[bs*i:bs*(i+1)])
        
    with open(os.path.join(target_path, fname[1]+'%05d.npz'%i), 'wb') as f_grasp:
        np.savez(f_grasp, hand_poses[bs*i:bs*(i+1)])
        
    with open(os.path.join(target_path, fname[2]+'%05d.npz'%i), 'wb') as f_metric:
        np.savez(f_metric, metric[bs*i:bs*(i+1)])
