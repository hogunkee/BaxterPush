import numpy as np

class History:
  def __init__(self, config):
    self.cnn_format = config.cnn_format

    batch_size, history_length = config.batch_size, config.history_length
    screen_height, screen_width, screen_channel = config.screen_height, config.screen_width, config.screen_channel
    # screen_height, screen_width, screen_channel = 256, 256, 4
    # batch_size, history_length, screen_height, screen_width = \
    #     config.batch_size, config.history_length, config.screen_height, config.screen_width

    self.history = np.zeros(
      [history_length, 2, screen_height, screen_width, screen_channel], dtype=np.float32)
    # self.history = np.zeros(
    #     [history_length, screen_height, screen_width], dtype=np.float32)

  def add(self, screen):
    new_screen = np.array(screen)
    self.history[:-1] = self.history[1:]
    self.history[-1] = new_screen

  def reset(self):
    self.history *= 0

  def get(self):
    if self.cnn_format == 'NHWC':
      return np.transpose(self.history, (1, 2, 0))
    else:
      return self.history
