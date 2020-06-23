from __future__ import print_function
import random
import tensorflow as tf

from dqn.agent import Agent
from dqn.environment import GymEnvironment, SimpleGymEnvironment
from config import get_config

import sys
sys.path.append('/home/scarab5/hogun_codes/robosuite-gqcnn/robosuite/robosuite/scripts')
from robosuite.scripts.demo_baxter_rl_pushing import *

flags = tf.app.flags

# Model
flags.DEFINE_string('model', 'm1', 'Type of model')
flags.DEFINE_boolean('dueling', False, 'Whether to use dueling deep q-network')
flags.DEFINE_boolean('double_q', False, 'Whether to use double q-learning')

# Environment
#flags.DEFINE_string('env_name', 'Breakout-v0', 'The name of gym environment to use')
flags.DEFINE_integer('action_repeat', 1, 'The number of action to be repeated')

# Etc
flags.DEFINE_boolean('use_gpu', False, 'Whether to use gpu or not') # True
flags.DEFINE_string('gpu_fraction', '1/1', 'idx / # of gpu fraction e.g. 1/3, 2/3, 3/3')
flags.DEFINE_boolean('display', False, 'Whether to do display the game screen or not')
flags.DEFINE_boolean('is_train', True, 'Whether to do training or testing')
flags.DEFINE_integer('random_seed', 123, 'Value of random seed')

FLAGS = flags.FLAGS

# Set random seed
tf.set_random_seed(FLAGS.random_seed)
random.seed(FLAGS.random_seed)

if FLAGS.gpu_fraction == '':
  raise ValueError("--gpu_fraction should be defined")

def calc_gpu_fraction(fraction_string):
  idx, num = fraction_string.split('/')
  idx, num = float(idx), float(num)

  fraction = 1 / (num - idx + 1)
  print(" [*] GPU : %.4f" % fraction)
  return fraction

def main(_):
  gpu_options = tf.GPUOptions(
      per_process_gpu_memory_fraction=calc_gpu_fraction(FLAGS.gpu_fraction))

  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--seed', type=int, default=0)
  parser.add_argument(
      '--num-objects', type=int, default=2)
  parser.add_argument(
      '--num-episodes', type=int, default=10000)
  parser.add_argument(
      '--num-steps', type=int, default=1)
  parser.add_argument(
      '--render', type=bool, default=False) #True
  parser.add_argument(
      '--bin-type', type=str, default="table")  # table, bin, two
  parser.add_argument(
      '--object-type', type=str, default="cube")  # T, Tlarge, L, 3DNet, stick, round_T_large
  parser.add_argument(
      '--test', type=bool, default=False)
  parser.add_argument(
      '--config-file', type=str, default="config_example.yaml")
  args = parser.parse_args()

  with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    config = get_config(FLAGS) or FLAGS

    env = robosuite.make(
        "BaxterPush",
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
    env = BaxterEnv(env, task='push', render=args.render)

    if not tf.test.is_gpu_available() and FLAGS.use_gpu:
      raise Exception("use_gpu flag is true when no GPUs are available")

    if not FLAGS.use_gpu:
      config.cnn_format = 'NHWC'

    agent = Agent(config, env, sess)

    if FLAGS.is_train:
      agent.train()
    else:
      agent.play()

if __name__ == '__main__':
  tf.app.run()