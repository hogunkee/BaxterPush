class AgentConfig(object):
  scale = 200 #10000
  display = False

  max_step = 500000 #5000 * scale
  memory_size = 10000 #100 * scale

  batch_size = 512 #128
  random_start = 30
  cnn_format = 'NHWC' #'NCHW'
  discount = 0.99
  target_q_update_step = 200 #1 * scale
  learning_rate = 1e-4 #0.00025 #1e-4 #
  learning_rate_minimum = 1e-6 #0.00025 #1e-6
  learning_rate_decay = 0.96
  learning_rate_decay_step = 1000 #5 * scale

  ep_end = 0.1
  ep_start = 1.
  ep_end_t = memory_size

  # history_length = 4
  train_frequency = 4
  learn_start = 1000 # 1. * scale # 5.

  min_delta = -1
  max_delta = 1

  double_q = True #False
  dueling = True #False

  _test_step = 500 # 5 * scale
  _save_step = 2000 # _test_step * 10

class EnvironmentConfig(object):
  env_name = 'BaxterPush'

  crop = None
  screen_width  = 128 #64 #84 #256
  screen_height = 128 #64 #84 #256
  screen_channel = 3

  max_reward = 1.
  min_reward = -1.

class DQNConfig(AgentConfig, EnvironmentConfig):
  model = ''
  pass

class M1(DQNConfig):
  backend = 'tf'
  env_type = 'detail'
  action_repeat = 1
  model_name = None

def get_config(FLAGS):
  if FLAGS.model == 'm1':
    config = M1
  elif FLAGS.model == 'm2':
    config = M2

  config.cnn_format = 'NHWC'
  # for k in FLAGS.__dict__['__wrapped']:
  #   if k == 'use_gpu':
  #     if not FLAGS.__getattr__(k):
  #       config.cnn_format = 'NHWC'
  #     else:
  #       config.cnn_format = 'NCHW'
  for k in FLAGS.__dict__['__wrapped']:
    if hasattr(config, k):
      setattr(config, k, FLAGS.__getattr__(k))

  # for k, v in FLAGS.__dict__['__flags'].items():
  #   if k == 'gpu':
  #     if v == False:
  #       config.cnn_format = 'NHWC'
  #     else:
  #       config.cnn_format = 'NCHW'
  #
  #   if hasattr(config, k):
  #     setattr(config, k, v)

  return config
