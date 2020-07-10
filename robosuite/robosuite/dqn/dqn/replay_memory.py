"""Code from https://github.com/tambetm/simple_dqn/blob/master/src/replay_memory.py"""

import os
import random
import logging
import numpy as np

from .utils import save_npy, load_npy

class ReplayMemory:
  def __init__(self, config, model_dir, using_feature=False):
    self.model_dir = model_dir

    self.cnn_format = config.cnn_format
    self.memory_size = config.memory_size
    self.actions = np.empty(self.memory_size, dtype = np.uint8)
    self.rewards = np.empty(self.memory_size, dtype = np.integer)
    if using_feature:
      self.screens = np.empty((self.memory_size, 9))
      self.dims = (9,)
    else:
      self.screens = np.empty((self.memory_size, 2, config.screen_height, config.screen_width, config.screen_channel), dtype = np.float16)
      self.dims = (2, config.screen_height, config.screen_width, config.screen_channel)
    self.terminals = np.empty(self.memory_size, dtype = np.bool)
    # self.history_length = config.history_length
    self.batch_size = config.batch_size
    self.count = 0
    self.current = 0

    # pre-allocate prestates and poststates for minibatch
    self.prestates = np.empty((self.batch_size,) + self.dims, dtype=np.float16)
    self.poststates = np.empty((self.batch_size,) + self.dims, dtype=np.float16)
    # self.prestates = np.empty((self.batch_size, self.history_length) + self.dims, dtype = np.float16)
    # self.poststates = np.empty((self.batch_size, self.history_length) + self.dims, dtype = np.float16)

  def add(self, screen, reward, action, terminal):
    assert screen.shape == self.dims
    # NB! screen is post-state, after action and reward
    self.actions[self.current] = action
    self.rewards[self.current] = reward
    self.screens[self.current, ...] = screen
    self.terminals[self.current] = terminal
    self.count = max(self.count, self.current + 1)
    self.current = (self.current + 1) % self.memory_size

  def getState(self, index):
    assert self.count > 0, "replay memory is empty, use at least --random_steps 1"
    # normalize index to expected range, allows negative indexes
    index = index % self.count
    # if is not in the beginning of matrix
    return self.screens[index]
    '''
    if index >= self.history_length - 1:
      # use faster slicing
      return self.screens[(index - (self.history_length - 1)):(index + 1), ...]
    else:
      # otherwise normalize indexes and use slower list based access
      indexes = [(index - i) % self.count for i in reversed(range(self.history_length))]
      return self.screens[indexes, ...]
    '''

  def sample(self):
    # memory must include poststate, prestate and history
    # assert self.count > self.history_length
    # sample random indexes
    indexes = []
    while len(indexes) < self.batch_size:
      # find random index 
      while True:
        # sample one index (ignore states wraping over 
        index = random.randint(0, self.count - 1)
        # if wraps over current pointer, then get new one
        # if index >= self.current: # and index - self.history_length < self.current:
        #  continue
        # if wraps over episode end, then get new one
        # NB! poststate (last screen) can be terminal state!
        # if self.terminals[(index - self.history_length):index].any():
        if self.terminals[index]:
          continue
        # otherwise use this index
        break
      
      # NB! having index first is fastest in C-order matrices
      self.prestates[len(indexes), ...] = self.getState(index - 1)
      self.poststates[len(indexes), ...] = self.getState(index)
      indexes.append(index)

    actions = self.actions[indexes]
    rewards = self.rewards[indexes]
    terminals = self.terminals[indexes]

    return self.prestates, actions, rewards, self.poststates, terminals
    # if self.cnn_format == 'NHWC':
    #   return np.transpose(self.prestates, (0, 1, 3, 4, 2)), actions, \
    #     rewards, np.transpose(self.poststates, (0, 1, 3, 4, 2)), terminals
    # else:
    #   return self.prestates, actions, rewards, self.poststates, terminals

  def save(self):
    for idx, (name, array) in enumerate(
        zip(['actions', 'rewards', 'screens', 'terminals', 'prestates', 'poststates'],
            [self.actions, self.rewards, self.screens, self.terminals, self.prestates, self.poststates])):
      save_npy(array, os.path.join(self.model_dir, name))

  def load(self):
    for idx, (name, array) in enumerate(
        zip(['actions', 'rewards', 'screens', 'terminals', 'prestates', 'poststates'],
            [self.actions, self.rewards, self.screens, self.terminals, self.prestates, self.poststates])):
      array = load_npy(os.path.join(self.model_dir, name))
