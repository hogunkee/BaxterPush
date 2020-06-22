# -*- coding: utf-8 -*-
"""
Copyright ©2017. The Regents of the University of California (Regents). All Rights Reserved.
Permission to use, copy, modify, and distribute this software and its documentation for educational,
research, and not-for-profit purposes, without fee and without a signed licensing agreement, is
hereby granted, provided that the above copyright notice, this paragraph and the following two
paragraphs appear in all copies, modifications, and distributions. Contact The Office of Technology
Licensing, UC Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-
7201, otl@berkeley.edu, http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.

IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF
THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED
HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE
MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
"""
"""
Displays robust grasps planned using a GQ-CNN-based policy on a set of saved RGB-D images.
The default configuration for the standard GQ-CNN policy is cfg/examples/policy.yaml. The default configuration for the Fully-Convolutional GQ-CNN policy is cfg/examples/fc_policy.yaml.

Author
------
Jeff Mahler and Vishal Satish
"""
import cv2
import json
import os
import time

import numpy as np

from autolab_core import RigidTransform, YamlConfig, Logger
from perception import BinaryImage, CameraIntrinsics, ColorImage, DepthImage, RgbdImage
from visualization import Visualizer2D as vis

from gqcnn.grasping import RobustGraspingPolicy, CrossEntropyRobustGraspingPolicy, RgbdImageState, FullyConvolutionalGraspingPolicyParallelJaw, FullyConvolutionalGraspingPolicySuction
from gqcnn.utils import GripperMode, NoValidGraspsException

from matplotlib import pyplot as plt

# set up logger
logger = Logger.get_logger('examples/policy_mujoco.py')

def evaluate_gqcnn(color_image, depth_image, normalize=False, vis_on=True, model_name='GQ-Image-Wise', crop_size=32):
    #model_name = 'GQ-Image-Wise' 
    #model_name = 'GQ-Bin-Picking-Eps90'
    #model_name = 'FC-GQCNN-4.0-PJ'
    segmask_filename = None
    model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),'../../../gqcnn/models')
    camera_intr_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),\
            'primesense.intr')
    config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'gqcnn_pj.yaml')
    fully_conv = False 

    assert not (fully_conv and depth_image is not None and segmask_filename is None), 'Fully-Convolutional policy expects a segmask.'

    # set model if provided 
    model_path = os.path.join(model_dir, model_name)

    # get configs
    model_config = json.load(open(os.path.join(model_path, 'config.json'), 'r'))
    try:
        gqcnn_config = model_config['gqcnn']
        gripper_mode = gqcnn_config['gripper_mode']
    except:
        gqcnn_config = model_config['gqcnn_config']
        input_data_mode = gqcnn_config['input_data_mode']
        if input_data_mode == 'tf_image':
            gripper_mode = GripperMode.LEGACY_PARALLEL_JAW
        elif input_data_mode == 'tf_image_suction':
            gripper_mode = GripperMode.LEGACY_SUCTION                
        elif input_data_mode == 'suction':
            gripper_mode = GripperMode.SUCTION                
        elif input_data_mode == 'multi_suction':
            gripper_mode = GripperMode.MULTI_SUCTION                
        elif input_data_mode == 'parallel_jaw':
            gripper_mode = GripperMode.PARALLEL_JAW
        else:
            raise ValueError('Input data mode {} not supported!'.format(input_data_mode))
    
    # set config
    if config_filename is None:
        if gripper_mode == GripperMode.LEGACY_PARALLEL_JAW or gripper_mode == GripperMode.PARALLEL_JAW:
            if fully_conv:
                config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                               '..',
                                               'cfg/examples/fc_gqcnn_pj.yaml')
            else:
                config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                               '..',
                                               'cfg/examples/gqcnn_pj.yaml')
        elif gripper_mode == GripperMode.LEGACY_SUCTION or gripper_mode == GripperMode.SUCTION:
            if fully_conv:
                config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                               '..',
                                               'cfg/examples/fc_gqcnn_suction.yaml')
            else:
                config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                               '..',
                                               'cfg/examples/gqcnn_suction.yaml')
            
    # read config
    config = YamlConfig(config_filename)
    inpaint_rescale_factor = config['inpaint_rescale_factor']
    policy_config = config['policy']

    # make relative paths absolute
    if 'gqcnn_model' in policy_config['metric'].keys():
        policy_config['metric']['gqcnn_model'] = model_path
        if not os.path.isabs(policy_config['metric']['gqcnn_model']):
            policy_config['metric']['gqcnn_model'] = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', policy_config['metric']['gqcnn_model'])
            
    # setup sensor
    camera_intr = CameraIntrinsics.load(camera_intr_filename)
        
    # read images
    if normalize:
        y1 = 0.5
        y2 = 0.8
        min_d = np.min(depth_image)
        max_d = np.max(depth_image)
        depth_image = (y2-y1)/(max_d-min_d)*depth_image + y1 - min_d*(y2-y1)/(max_d-min_d)
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('max depth: %.3f', np.max(depth_image))
        print('min depth: %.3f', np.min(depth_image))
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        if np.isnan(np.max(depth_image)): # check camera broken
            return None, None

    if np.min(depth_image) > 0.99:
        return None, None

    depth_im = DepthImage(depth_image, frame=camera_intr.frame)
    color_im = ColorImage(color_image, frame=camera_intr.frame)
    #color_im = ColorImage(np.zeros([depth_im.height, depth_im.width, 3]).astype(np.uint8),
    #                      frame=camera_intr.frame)
    
    # optionally read a segmask
    segmask = None
    if segmask_filename is not None:
        segmask = BinaryImage.open(segmask_filename)
    valid_px_mask = depth_im.invalid_pixel_mask().inverse()
    print(valid_px_mask)
    print(dir(valid_px_mask))
    if segmask is None:
        segmask = valid_px_mask
    else:
        segmask = segmask.mask_binary(valid_px_mask)
    
    # inpaint
    depth_im = depth_im.inpaint(rescale_factor=inpaint_rescale_factor)
        
    if 'input_images' in policy_config['vis'].keys() and policy_config['vis']['input_images']:
        vis.figure(size=(10,10))
        num_plot = 1
        if segmask is not None:
            num_plot = 2
        vis.subplot(1,num_plot,1)
        vis.imshow(depth_im)
        if segmask is not None:
            vis.subplot(1,num_plot,2)
            vis.imshow(segmask)
        vis.show()
        

    # create state
    rgbd_im = RgbdImage.from_color_and_depth(color_im, depth_im)
    state = RgbdImageState(rgbd_im, camera_intr, segmask=segmask)

    # set input sizes for fully-convolutional policy
    if fully_conv:
        policy_config['metric']['fully_conv_gqcnn_config']['im_height'] = depth_im.shape[0]
        policy_config['metric']['fully_conv_gqcnn_config']['im_width'] = depth_im.shape[1]

    # init policy
    if fully_conv:
        #TODO: @Vishal we should really be doing this in some factory policy
        if policy_config['type'] == 'fully_conv_suction':
            policy = FullyConvolutionalGraspingPolicySuction(policy_config)
        elif policy_config['type'] == 'fully_conv_pj':
            policy = FullyConvolutionalGraspingPolicyParallelJaw(policy_config)
        else:
            raise ValueError('Invalid fully-convolutional policy type: {}'.format(policy_config['type']))
    else:
        policy_type = 'cem'
        if 'type' in policy_config.keys():
            policy_type = policy_config['type']
        if policy_type == 'ranking':
            policy = RobustGraspingPolicy(policy_config)
        elif policy_type == 'cem':
            policy = CrossEntropyRobustGraspingPolicy(policy_config)
        else:
            raise ValueError('Invalid policy type: {}'.format(policy_type))

    # query policy
    policy_start = time.time()
    action = policy(state)
    logger.info('Planning took %.3f sec' %(time.time() - policy_start))

    # vis final grasp
    if vis_on and policy_config['vis']['final_grasp']:
        vis.figure(size=(10,10))
        '''
        vis.imshow(rgbd_im.depth)
        '''
        vis.imshow(rgbd_im.depth,
                   vmin=policy_config['vis']['vmin'],
                   vmax=policy_config['vis']['vmax'])
        vis.grasp(action.grasp, scale=2.5, show_center=False, show_axis=True)
        vis.title('Planned grasp at depth {0:.3f}m with Q={1:.3f}'.format(action.grasp.depth, action.q_value))
        vis.show()

        vis.imshow(rgbd_im.color,
                   vmin=policy_config['vis']['vmin'],
                   vmax=policy_config['vis']['vmax'])
        vis.grasp(action.grasp, scale=2.5, show_center=False, show_axis=True)
        vis.title('Planned grasp at depth {0:.3f}m with Q={1:.3f}'.format(action.grasp.depth, action.q_value))
        vis.show()


    def getTranslationMatrix2d(dx, dy):
        """
        Returns a numpy affine transformation matrix for a 2D translation of
        (dx, dy)
        """
        return np.matrix([[1, 0, dx], [0, 1, dy], [0, 0, 1]])


    def rotateImage(image, cx, cy, angle):
        """
        Rotates the given image about it's centre
        """

        image_size = (image.shape[1], image.shape[0])
        image_center = tuple(np.array([cx, cy]))

        rot_mat = np.vstack([cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]])

        new_image_size = (crop_size*2, crop_size*2) #(64, 64)

        new_midx, new_midy = crop_size, crop_size #32, 32 

        dx = int(new_midx - cx)
        dy = int(new_midy - cy)

        trans_mat = getTranslationMatrix2d(dx, dy)
        affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]
        result = cv2.warpAffine(np.float32(image.data), affine_mat, new_image_size)

        return result

    '''
    vis.figure(size=(10,10))
    vis.imshow(rgbd_im.depth,
               vmin=policy_config['vis']['vmin'],
               vmax=policy_config['vis']['vmax'])
    vis.grasp(action.grasp, scale=2.5, show_center=False, show_axis=True)
    vis.title('Planned grasp at depth {0:.3f}m with Q={1:.3f}'.format(action.grasp.depth, action.q_value))
    #vis.savefig('depth2.png')
    vis.show()
    '''

    depth_im = rgbd_im.depth
    cx, cy = action.grasp.center
    angle = action.grasp.angle *180/np.pi

    # rotate and crop and downsample
    if crop_size==96:
        cropped_im = cv2.resize(rotateImage(depth_im.data, cx, cy, 1e-6), (crop_size, crop_size)) #(32, 32)
    elif crop_size==32:
        cropped_im = cv2.resize(rotateImage(depth_im.data, cx, cy, angle), (crop_size, crop_size)) #(32, 32)
    print('angle:', angle)

    '''
    vis.figure(size=(10,10))
    vis.imshow(DepthImage(cropped_im),
           vmin=policy_config['vis']['vmin'],
           vmax=policy_config['vis']['vmax'])
    vis.title('cropped grasp image')
    vis.show()
    '''

    return action.grasp, cropped_im
