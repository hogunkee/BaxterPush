class AgentConfig(object):
  scale = 100 #10000
  display = False

  max_step = 5000 * scale
  memory_size = 100 * scale

  batch_size = 32
  random_start = 30
  cnn_format = 'NHWC' #'NCHW'
  discount = 0.99
  target_q_update_step = 1 * scale
  learning_rate = 1e-4 #0.00025
  learning_rate_minimum = 1e-6 #0.00025
  learning_rate_decay = 0.96
  learning_rate_decay_step = 5 * scale

  ep_end = 0.1
  ep_start = 1.
  ep_end_t = memory_size

  # history_length = 4
  train_frequency = 4
  learn_start = 5. * scale

  min_delta = -1
  max_delta = 1

  double_q = False
  dueling = False

  _test_step = 5 * scale
  _save_step = _test_step * 10

class EnvironmentConfig(object):
  env_name = 'BaxterPush'

  screen_width  = 256
  screen_height = 256
  screen_channel = 4
  max_reward = 1.
  min_reward = -1.

class DQNConfig(AgentConfig, EnvironmentConfig):
  model = ''
  pass

class M1(DQNConfig):
  backend = 'tf'
  env_type = 'detail'
  action_repeat = 1

def get_config(FLAGS):
  if FLAGS.model == 'm1':
    config = M1
  elif FLAGS.model == 'm2':
    config = M2

  for k in FLAGS.__dict__['__wrapped']:
    if k == 'use_gpu':
      if not FLAGS.__getattr__(k):
        config.cnn_format = 'NHWC'
      else:
        config.cnn_format = 'NCHW'

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
