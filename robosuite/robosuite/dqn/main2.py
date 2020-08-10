from __future__ import print_function
import random
import tensorflow as tf

from dqn.agent import Agent, ResNetAgent
from dqn.environment import GymEnvironment, SimpleGymEnvironment
from config import get_config

import os
import sys
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '../..'))
sys.path.append(os.path.join(FILE_PATH, '..', 'scripts'))
from demo_baxter_rl_pushing import *

flags = tf.app.flags

# Model
flags.DEFINE_string('model', 'm1', 'Type of model')
flags.DEFINE_boolean('dueling', True, 'Whether to use dueling deep q-network')
flags.DEFINE_boolean('double_q', True, 'Whether to use double q-learning')
flags.DEFINE_integer('cutout', 0, 'random cutout - with prob 0.7, 5~20 pixels (default)')

# Environment
#flags.DEFINE_string('env_name', 'Breakout-v0', 'The name of gym environment to use')
flags.DEFINE_integer('action_repeat', 1, 'The number of action to be repeated')
flags.DEFINE_string('task', 'reach', 'name of task: reach / push / pick')

# Etc
flags.DEFINE_boolean('use_gpu', True, 'Whether to use gpu or not') # True
flags.DEFINE_integer('resnet', 1, 'use ResNet or basic CNN')
# flags.DEFINE_string('gpu_fraction', '1/1', 'idx / # of gpu fraction e.g. 1/3, 2/3, 3/3')
flags.DEFINE_boolean('display', False, 'Whether to do display the game screen or not')
flags.DEFINE_boolean('is_train', True, 'Whether to do training or testing')
flags.DEFINE_integer('random_seed', 123, 'Value of random seed')

# baxter configuration
flags.DEFINE_integer('seed', 0, 'random seed')
flags.DEFINE_integer('num_objects', 2, 'number of objects')
flags.DEFINE_integer('num_episodes', 10000, 'number of episodes')
flags.DEFINE_integer('num_steps', 1, 'number of steps')
flags.DEFINE_integer('render', 0, 'Whether to do rendering or not')
flags.DEFINE_integer('using_feature', 0, 'Whether to use feature states or images')
flags.DEFINE_integer('using_rgbd', 0, 'Whether to use rgb-d images or rgb images')
flags.DEFINE_string('bin_type', 'table', 'bin type')
flags.DEFINE_string('object_type', 'cube', 'object type')
flags.DEFINE_integer('test', 0, 'Test or not')
flags.DEFINE_string('config_file', 'config_example.yaml', 'config file name')
flags.DEFINE_string('model_name', None, 'name of trained model') #'0707_140753', '0707_145027'

FLAGS = flags.FLAGS

render = bool(FLAGS.render)
using_feature = bool(FLAGS.using_feature)
test = bool(FLAGS.test)
cutout = bool(FLAGS.cutout)
using_rgbd = bool(FLAGS.using_rgbd)
using_resnet = bool(FLAGS.resnet)

# Set random seed
tf.set_random_seed(FLAGS.random_seed)
random.seed(FLAGS.random_seed)

# if FLAGS.gpu_fraction == '':
#   raise ValueError("--gpu_fraction should be defined")

def calc_gpu_fraction(fraction_string):
  idx, num = fraction_string.split('/')
  idx, num = float(idx), float(num)

  fraction = 1 / (num - idx + 1)
  print(" [*] GPU : %.4f" % fraction)
  return fraction

def main(_):
  gpu_config = tf.ConfigProto()
  gpu_config.gpu_options.allow_growth = True
  with tf.Session(config=gpu_config) as sess:
    config = get_config(FLAGS) or FLAGS

    env = robosuite.make(
        "BaxterPush",
        bin_type=FLAGS.bin_type,
        object_type=FLAGS.object_type,
        ignore_done=True,
        has_renderer=True,
        camera_name="eye_on_right_wrist",
        gripper_visualization=False,
        use_camera_obs=False,
        use_object_obs=False,
        camera_depth=True,
        num_objects=FLAGS.num_objects,
        control_freq=100,
        camera_width=config.screen_width,
        camera_height=config.screen_height,
        crop=config.crop
    )
    env = IKWrapper(env)
    env = BaxterEnv(env, task=FLAGS.task, render=render, using_feature=using_feature, rgbd=using_rgbd, print_on=test)

    if not tf.test.is_gpu_available() and FLAGS.use_gpu:
      raise Exception("use_gpu flag is true when no GPUs are available")

    if not FLAGS.use_gpu:
      config.cnn_format = 'NHWC'

    if using_resnet:
      agent = ResNetAgent(config, env, FLAGS.task, sess, cutout=cutout, rgbd=using_rgbd)
    else:
      agent = Agent(config, env, FLAGS.task, sess, cutout=cutout, rgbd=using_rgbd)

    if FLAGS.is_train and not test:
      agent.train()
    else:
      agent.play()

if __name__ == '__main__':
  tf.app.run()
