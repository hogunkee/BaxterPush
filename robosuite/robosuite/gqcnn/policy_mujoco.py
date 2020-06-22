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
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
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

# set up logger
logger = Logger.get_logger('examples/policy_mujoco.py')

#model_name = 'GQ-Image-Wise' 
#model_name = 'GQ-Bin-Picking-Eps90'
#model_name = 'FC-GQCNN-4.0-PJ'
#model_name = '6D-FC-GQCNN'

class GQCNN():
    def __init__(self, model_ver=0, model_name=None):
        self.model_ver = model_ver
        if self.model_ver==0 or self.model_ver==1:
            self.crop_size = 32
        elif self.model_ver==2:
            self.crop_size = 96
        elif self.model_ver==3:
            self.crop_size = 100
        elif self.model_ver == 4:
            self.crop_size = 100
        else:
            print('Wrong model version.')
            raise ValueError('Invalid GQ-CNN model version: %d'%model_ver)
            exit()

        self.fully_conv = False
        self.model_name = model_name
        if self.model_name is None:
            if self.model_ver==0:
                self.model_name = 'GQ-Image-Wise'
            elif self.model_ver==1:
                self.model_name = 'GQ-Bin-Picking-Eps90'
            elif self.model_ver==2:
                self.model_name = 'FC-GQCNN-4.0-PJ'
                self.fully_conv = True
            elif self.model_ver==3:
                self.model_name = '6D-FC-GQCNN'
                self.fully_conv = True
            elif self.model_ver == 4:
                self.model_name = '6D-QNet'
                self.fully_conv = True

        self._load_model()
        if model_ver == 4:
            self.evaluate_gqcnn = self.evalute_6Dqnet
        elif model_ver!=3:
            self.evaluate_gqcnn = self.evaluate_4Dgqcnn
        else:
            self.evaluate_gqcnn = self.evaluate_6Dgqcnn


    def evaluate_4Dgqcnn(self, color_image, depth_image, vis_on=True, num_candidates=1):
        # setup sensor
        camera_intr_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),\
                'primesense.intr')
        camera_intr = CameraIntrinsics.load(camera_intr_filename)

        # check camera broken
        if np.min(depth_image) > 0.99:
            return None, None, None

        depth_im = DepthImage(depth_image, frame=camera_intr.frame)
        color_im = ColorImage(color_image, frame=camera_intr.frame)
        
        
        # inpaint
        depth_im = depth_im.inpaint(rescale_factor=self.inpaint_rescale_factor)
            
        if 'input_images' in self.policy_config['vis'].keys() and self.policy_config['vis']['input_images']:
            vis.figure(size=(10,10))
            num_plot = 1
            vis.subplot(1,num_plot,1)
            vis.imshow(depth_im)
            vis.show()
            

        #segmask
        valid_px_mask = depth_im.invalid_pixel_mask().inverse()
        # print(valid_px_mask.data)
        # plt.imshow(valid_px_mask.data)
        # plt.show()
        # print(dir(valid_px_mask))
        segmask = valid_px_mask

        # create state
        rgbd_im = RgbdImage.from_color_and_depth(color_im, depth_im)
        state = RgbdImageState(rgbd_im, camera_intr, segmask=segmask)
        '''
        # set input sizes for fully-convolutional policy
        if fully_conv:
            self.policy_config['metric']['fully_conv_gqcnn_config']['im_height'] = depth_im.shape[0]
            self.policy_config['metric']['fully_conv_gqcnn_config']['im_width'] = depth_im.shape[1]
        '''

        # query policy
        policy_start = time.time()

        if self.model_ver == 2:
            num_actions = num_candidates
            actions = self.policy(state, num_actions) ################ crucial part
            if num_actions == 1:
                action = actions
            else:
                action = actions[0]
        else:
            actions = self.policy(state)
            action = actions
            
        
        logger.info('Planning took %.3f sec' %(time.time() - policy_start))

        # vis final grasp
        '''if vis_on and self.policy_config['vis']['final_grasp']:
            vis.figure(size=(10,10))
            vis.imshow(rgbd_im.depth,
                       vmin=self.policy_config['vis']['vmin'],
                       vmax=self.policy_config['vis']['vmax'])
            vis.grasp(action.grasp, scale=2.5, show_center=False, show_axis=True)
            vis.title('Planned grasp at depth {0:.3f}m with Q={1:.3f}'.format(action.grasp.depth, action.q_value))
            vis.show()

            vis.imshow(rgbd_im.color,
                       vmin=self.policy_config['vis']['vmin'],
                       vmax=self.policy_config['vis']['vmax'])
            vis.grasp(action.grasp, scale=2.5, show_center=False, show_axis=True)
            vis.title('Planned grasp at depth {0:.3f}m with Q={1:.3f}'.format(action.grasp.depth, action.q_value))
            vis.show()'''

        if vis_on and self.policy_config['vis']['final_grasp']:
            vis.figure(size=(10,10))
            '''
            vis.imshow(rgbd_im.depth)
            '''
            vis.imshow(rgbd_im.depth,
                       vmin=self.policy_config['vis']['vmin'],
                       vmax=self.policy_config['vis']['vmax'])
            for i in range(num_candidates):
                vis.grasp(actions[i].grasp, scale=2.5, show_center=False, show_axis=True)
            vis.show()

            vis.imshow(rgbd_im.color,
                       vmin=self.policy_config['vis']['vmin'],
                       vmax=self.policy_config['vis']['vmax'])
            for i in range(num_candidates):
                vis.grasp(actions[i].grasp, scale=2.5, show_center=False, show_axis=True)
            vis.show()


        def getTranslationMatrix2d(dx, dy):
            """
            Returns a numpy affine transformation matrix for a 2D translation of
            (dx, dy)
            """
            return np.matrix([[1, 0, dx], [0, 1, dy], [0, 0, 1]])


        def rotateImage(image, cx, cy, angle):
            """
            Rotates the given image about it's center
            """
            image_size = (image.shape[1], image.shape[0])
            image_center = tuple(np.array([cx, cy]))

            rot_mat = np.vstack([cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]])

            new_image_size = (self.crop_size*2, self.crop_size*2) #(64, 64)

            new_midx, new_midy = self.crop_size, self.crop_size #32, 32 

            dx = int(new_midx - cx)
            dy = int(new_midy - cy)

            trans_mat = getTranslationMatrix2d(dx, dy)
            affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]
            result = cv2.warpAffine(np.float32(image.data), affine_mat, new_image_size)

            return result


        depth_im = rgbd_im.depth
        cx, cy = action.grasp.center
        angle = action.grasp.angle *180/np.pi

        # rotate and crop and downsample
        if self.crop_size==96:
            cropped_im = cv2.resize(rotateImage(depth_im.data, cx, cy, 1e-6), (self.crop_size, self.crop_size)) #(32, 32)
        elif self.crop_size==32:
            cropped_im = cv2.resize(rotateImage(depth_im.data, cx, cy, angle), (self.crop_size, self.crop_size)) #(32, 32)
        # print('angle:', angle)

        return actions, -1, cropped_im # cropped_im
        # return action.grasp, -1, cropped_im

    def evaluate_4Dgqcnn_batch(self, color_image, depth_image, vis_on=True, num_candidates=1, num_vps=4):
        # setup sensor
        camera_intr_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),\
                'primesense.intr')
        camera_intr = CameraIntrinsics.load(camera_intr_filename)

        # check camera broken
        if np.min(depth_image) > 0.99:
            return None, None, None
        
        states = []
        # depth_ims = []
        color_im = ColorImage(color_image, frame=camera_intr.frame) 
        # color_ims = [color_im] * num_vps
        for k in range(num_vps):
            depth_im = DepthImage(depth_image[k,:,:], frame=camera_intr.frame) 
            # inpaint
            depth_im = depth_im.inpaint(rescale_factor=self.inpaint_rescale_factor)
            # depth_ims.append(depth_im) 

            if 'input_images' in self.policy_config['vis'].keys() and self.policy_config['vis']['input_images']:
                vis.figure(size=(10,10))
                num_plot = 1
                vis.subplot(1,num_plot,1)
                vis.imshow(depth_im)
                vis.show()

            #segmask
            valid_px_mask = depth_im.invalid_pixel_mask().inverse()
            # print(valid_px_mask.data)
            # plt.imshow(valid_px_mask.data)
            # plt.show()
            # print(dir(valid_px_mask))
            segmask = valid_px_mask

            # create state
            rgbd_im = RgbdImage.from_color_and_depth(color_im, depth_im)
            state = RgbdImageState(rgbd_im, camera_intr, segmask=segmask)
            states.append(state)
        '''
        # set input sizes for fully-convolutional policy
        if fully_conv:
            self.policy_config['metric']['fully_conv_gqcnn_config']['im_height'] = depth_im.shape[0]
            self.policy_confidg['metric']['fully_conv_gqcnn_config']['im_width'] = depth_im.shape[1]
        '''

        # query policy
        policy_start = time.time()

        if self.model_ver == 2:
            num_actions = num_candidates
            actions = self.policy._action_batch(states, num_actions, num_vps) ################ crucial part
            if num_actions == 1:
                action = actions[0]
            else:
                action = actions[0][0]
        else:
            actions = self.policy(state)
            action = actions
            
        
        logger.info('Planning took %.3f sec' %(time.time() - policy_start))

        # vis final grasp
        if vis_on and self.policy_config['vis']['final_grasp']:
            vis.figure(size=(10,10))
            '''
            vis.imshow(rgbd_im.depth)
            '''
            vis.imshow(rgbd_im.depth,
                       vmin=self.policy_config['vis']['vmin'],
                       vmax=self.policy_config['vis']['vmax'])
            vis.grasp(action.grasp, scale=2.5, show_center=False, show_axis=True)
            vis.title('Planned grasp at depth {0:.3f}m with Q={1:.3f}'.format(action.grasp.depth, action.q_value))
            vis.show()

            vis.imshow(rgbd_im.color,
                       vmin=self.policy_config['vis']['vmin'],
                       vmax=self.policy_config['vis']['vmax'])
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
            Rotates the given image about it's center
            """
            image_size = (image.shape[1], image.shape[0])
            image_center = tuple(np.array([cx, cy]))

            rot_mat = np.vstack([cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]])

            new_image_size = (self.crop_size*2, self.crop_size*2) #(64, 64)

            new_midx, new_midy = self.crop_size, self.crop_size #32, 32 

            dx = int(new_midx - cx)
            dy = int(new_midy - cy)

            trans_mat = getTranslationMatrix2d(dx, dy)
            affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]
            result = cv2.warpAffine(np.float32(image.data), affine_mat, new_image_size)

            return result


        depth_im = rgbd_im.depth
        cx, cy = action.grasp.center
        angle = action.grasp.angle *180/np.pi

        # rotate and crop and downsample
        if self.crop_size==96:
            cropped_im = cv2.resize(rotateImage(depth_im.data, cx, cy, 1e-6), (self.crop_size, self.crop_size)) #(32, 32)
        elif self.crop_size==32:
            cropped_im = cv2.resize(rotateImage(depth_im.data, cx, cy, angle), (self.crop_size, self.crop_size)) #(32, 32)
        # print('angle:', angle)

        return actions, -1, cropped_im # cropped_im
        # return action.grasp, -1, cropped_im

    def evaluate_6Dgqcnn(self, color_image, depth_image, vis_on=False):
        if (np.min(depth_image)>2.5): # check camera broken
            return None, None, None

        def sample_depths(raw_depth_im, depth_offset, num_depth_bins):
            """Sample depths from the raw depth image."""
            max_depth = np.max(raw_depth_im) + depth_offset

            # for sampling the min depth, we only sample from the portion of the depth image in the object segmask because sometimes the rim of the bin is not properly subtracted out of the depth image
            # raw_depth_im_segmented = np.ones_like(raw_depth_im)
            # raw_depth_im_segmented[np.where(raw_seg > 0)] = raw_depth_im[np.where(raw_seg > 0)]
            min_depth = np.min(raw_depth_im) + depth_offset
            max_depth = 0.8

            depth_bin_width = (max_depth - min_depth) / num_depth_bins
            depths = np.zeros((num_depth_bins, 1)) 
            for i in range(num_depth_bins):
                depths[i][0] = min_depth + (i * depth_bin_width + depth_bin_width / 2)
            return depths
     
        ####
        from .grasp_network import FCGQCNN, reset_graph
        DATA_DIR = os.path.dirname(os.path.realpath(__file__))
        #DATA_DIR = "/home/rllab/robosuite-gqcnn/robosuite/robosuite/gqcnn"
        angular_bins = [8,4,8]
        fcgqcnn = FCGQCNN(DATA_DIR, is_training=False, angular_bins=angular_bins)
        pretrained_dir = os.path.join(DATA_DIR, 'graspnet')
        fcgqcnn.load_pretrained_model(pretrained_dir)

        raw_im = depth_image #depth_im.data # cv2.resize(depth_im.data,(100,100))
        pose = sample_depths(raw_im, 0, 16)

        print(pose)

        policy_start = time.time()

        predictions = fcgqcnn.predict(np.expand_dims(np.tile(raw_im,(16,1,1)), axis=-1), pose)

        dim_x = predictions.shape[1]
        dim_y = predictions.shape[2]
        # print('dim_x: ', dim_x)
        # print('dim_y: ', dim_y)
        pred = np.squeeze( np.reshape(predictions, (16,dim_x,dim_y,8,4,8,2) )[:,:,:,:,:,:,1] )  # np.prod(angular_bins)
        ind_all = np.argwhere(pred == pred.max())
        inds = []
        for idx, val in enumerate(ind_all):
            if val[2]<4 or val[2]>35:
                inds.append(idx)
                # ind_filt = np.delete(ind_, idx, axis=0)
        ind_filt = np.delete(ind_all, inds, axis=0)
        print('ind_filt: ',  ind_filt)
        ind = ind_filt[0]

        # query policy
        # action = policy(state)
        logger.info('Planning took %.3f sec' %(time.time() - policy_start))

        return ind, pose, depth_image

    def evaluate_6Dqnet(self, color_image, depth_image, angle, vis_on=False):
        if (np.min(depth_image)>2.5): # check camera broken
            return None, None, None

        def sample_depths(raw_depth_im, depth_offset, num_depth_bins):
            """Sample depths from the raw depth image."""
            max_depth = np.max(raw_depth_im) + depth_offset

            # for sampling the min depth, we only sample from the portion of the depth image in the object segmask because sometimes the rim of the bin is not properly subtracted out of the depth image
            # raw_depth_im_segmented = np.ones_like(raw_depth_im)
            # raw_depth_im_segmented[np.where(raw_seg > 0)] = raw_depth_im[np.where(raw_seg > 0)]
            min_depth = np.min(raw_depth_im) + depth_offset
            max_depth = 0.8

            depth_bin_width = (max_depth - min_depth) / num_depth_bins
            depths = np.zeros((num_depth_bins, 1)) 
            for i in range(num_depth_bins):
                depths[i][0] = min_depth + (i * depth_bin_width + depth_bin_width / 2)
            return depths
     
        ####
        from .qualitynet import FCGQCNN, reset_graph
        #DATA_DIR = os.path.dirname(os.path.realpath(__file__))
        DATA_DIR = "/home/rllab/robosuite-gqcnn/robosuite/robosuite/gqcnn/qualitynet"
        angular_bins = 16
        fcgqcnn = FCGQCNN(DATA_DIR, is_training=False, length=256, width=256, angular_bins=angular_bins)
        pretrained_dir = os.path.join(DATA_DIR, 'model')
        fcgqcnn.load_pretrained_model(pretrained_dir)

        raw_im = depth_image #depth_im.data # cv2.resize(depth_im.data,(100,100))
        pose = sample_depths(raw_im, 0, 16)
        pose.append()

        angle = np.reshape(angle, (1, 2))
        angle = np.tile(angle, (16, 1))
        pose = np.concatenate((pose, angle), axis=1)

        policy_start = time.time()

        predictions = fcgqcnn.predict(np.expand_dims(np.tile(raw_im,(16, 1, 1)), axis=-1), pose)

        dim_x = predictions.shape[1]
        dim_y = predictions.shape[2]
        # print('dim_x: ', dim_x)
        # print('dim_y: ', dim_y)
        pred = np.squeeze(np.reshape(predictions, (16, dim_x, dim_y, 16 ,2))[... ,1])  # np.prod(angular_bins)
        ind_all = np.argwhere(pred == pred.max())
        inds = []
        for idx, val in enumerate(ind_all):
            if val[2]<4 or val[2]>35:
                inds.append(idx)
                # ind_filt = np.delete(ind_, idx, axis=0)
        ind_filt = np.delete(ind_all, inds, axis=0)
        print('ind_filt: ',  ind_filt)
        ind = ind_filt[0]

        # query policy
        # action = policy(state)
        logger.info('Planning took %.3f sec' %(time.time() - policy_start))

        return ind, pose, depth_image

    def _load_model(self):
        model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),'.')
        #model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),'../../../gqcnn/models')
        if self.fully_conv:
            config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'fc_gqcnn_pj.yaml')
        else:
            config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'gqcnn_pj.yaml')

        # set model if provided 
        model_path = os.path.join(model_dir, self.model_name)

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


        # read config
        config = YamlConfig(config_filename)
        self.inpaint_rescale_factor = config['inpaint_rescale_factor']
        self.policy_config = config['policy']

        # make relative paths absolute
        if 'gqcnn_model' in self.policy_config['metric'].keys():
            self.policy_config['metric']['gqcnn_model'] = model_path
            if not os.path.isabs(self.policy_config['metric']['gqcnn_model']):
                self.policy_config['metric']['gqcnn_model'] = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', self.policy_config['metric']['gqcnn_model'])
                
            

        # init policy
        if self.fully_conv:
            #TODO: @Vishal we should really be doing this in some factory policy
            if self.policy_config['type'] == 'fully_conv_suction':
                self.policy = FullyConvolutionalGraspingPolicySuction(self.policy_config)
            elif self.policy_config['type'] == 'fully_conv_pj':
                self.policy = FullyConvolutionalGraspingPolicyParallelJaw(self.policy_config)
            else:
                raise ValueError('Invalid fully-convolutional policy type: {}'.format(self.policy_config['type']))
        else:
            policy_type = 'cem'
            if 'type' in self.policy_config.keys():
                policy_type = self.policy_config['type']
            if policy_type == 'ranking':
                self.policy = RobustGraspingPolicy(self.policy_config)
            elif policy_type == 'cem':
                self.policy = CrossEntropyRobustGraspingPolicy(self.policy_config)
            else:
                raise ValueError('Invalid policy type: {}'.format(policy_type))
            

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

    depth_im = rgbd_im.depth
    cx, cy = action.grasp.center
    angle = action.grasp.angle *180/np.pi

    # rotate and crop and downsample
    if crop_size==96:
        cropped_im = cv2.resize(rotateImage(depth_im.data, cx, cy, 1e-6), (crop_size, crop_size)) #(32, 32)
    elif crop_size==32:
        cropped_im = cv2.resize(rotateImage(depth_im.data, cx, cy, angle), (crop_size, crop_size)) #(32, 32)
    print('angle:', angle)

    return action.grasp, cropped_im
