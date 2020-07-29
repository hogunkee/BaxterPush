import os
import numpy as np
import pickle
from ops import *


class SimpleCNN():
    def __init__(self, task, cnn_format='NHWC'):
        self.task = task
        if task=='reach':
            self.action_size = 10
        elif task=='push':
            self.action_size = 10
        elif task=='pick':
            self.action_size = 12

        self.cnn_format = cnn_format
        self.screen_heidht = 64
        self.screen_width = 64
        self.screen_channel = 4

        self.dueling = False

    def set_datapath(self, data_path):
        file_list = os.listdir(data_path)
        pkl_list = [f for f in file_list if 'pkl' in f and self.task in f]
        if len(pkl_list)==0:
            print('No pickle files exist. Wrong data path!!')
            return
        self.pkl_list = sorted([os.path.join(data_path, p) for p in pkl_list])
        self.a_list = [p for p in pkl_list if self.task+'_a_' in p]
        self.s_list = [p for p in pkl_list if self.task + '_s_' in p]
        assert len(self.a_list) == len(self.s_list)

    def build_net(self):
        self.w = {}
        self.t_w = {}

        #initializer = tf.contrib.layers.xavier_initializer()
        initializer = tf.truncated_normal_initializer(0, 0.1) #0.02)
        activation_fn = tf.nn.relu

        # training network
        with tf.variable_scope('prediction'):
            if self.cnn_format == 'NHWC':
                self.s_t = tf.placeholder('float32', [None, 2, self.screen_height, self.screen_width, self.screen_channel], name='s_t')
            else:
                self.s_t = tf.placeholder('float32', [None, 2, self.screen_channel, self.screen_height, self.screen_width], name='s_t')

            self.s_t_0 = self.s_t[:, 0, :, :]
            self.s_t_1 = self.s_t[:, 1, :, :]

            self.l1_0, self.w['l1_w0'], self.w['l1_b0'] = conv2d(self.s_t_0,
                32, [8, 8], [4, 4], initializer, activation_fn, self.cnn_format, name='l1_0')
            self.l2_0, self.w['l2_w0'], self.w['l2_b0'] = conv2d(self.l1_0,
                64, [4, 4], [2, 2], initializer, activation_fn, self.cnn_format, name='l2_0')
            self.l3_0, self.w['l3_w0'], self.w['l3_b0'] = conv2d(self.l2_0,
                64, [3, 3], [1, 1], initializer, activation_fn, self.cnn_format, name='l3_0')

            self.l1_1, self.w['l1_w1'], self.w['l1_b1'] = conv2d(self.s_t_1,
                 32, [8, 8], [4, 4], initializer, activation_fn, self.cnn_format, name='l1_1')
            self.l2_1, self.w['l2_w1'], self.w['l2_b1'] = conv2d(self.l1_1,
                 64, [4, 4], [2, 2], initializer, activation_fn, self.cnn_format, name='l2_1')
            self.l3_1, self.w['l3_w1'], self.w['l3_b1'] = conv2d(self.l2_1,
                 64, [3, 3], [1, 1], initializer, activation_fn, self.cnn_format, name='l3_1')

            shape = self.l3_0.get_shape().as_list()
            self.l3_flat_0 = tf.reshape(self.l3_0, [-1, reduce(lambda x, y: x * y, shape[1:])])
            self.l3_flat_1 = tf.reshape(self.l3_1, [-1, reduce(lambda x, y: x * y, shape[1:])])
            self.l3_flat = tf.concat([self.l3_flat_0, self.l3_flat_1], axis=1)

            if self.dueling:
                self.value_hid, self.w['l4_val_w'], self.w['l4_val_b'] = \
                    linear(self.l3_flat, 512, activation_fn=activation_fn, name='value_hid')

                self.adv_hid, self.w['l4_adv_w'], self.w['l4_adv_b'] = \
                    linear(self.l3_flat, 512, activation_fn=activation_fn, name='adv_hid')

                self.value, self.w['val_w_out'], self.w['val_w_b'] = \
                    linear(self.value_hid, 1, name='value_out')

                self.advantage, self.w['adv_w_out'], self.w['adv_w_b'] = \
                    linear(self.adv_hid, self.action_size, name='adv_out')

                # Average Dueling
                self.q = self.value + (self.advantage - tf.reduce_mean(self.advantage, reduction_indices=1, keep_dims=True))
            else:
                self.l4, self.w['l4_w'], self.w['l4_b'] = linear(self.l3_flat, 512, activation_fn=activation_fn, name='l4')
                self.q, self.w['q_w'], self.w['q_b'] = linear(self.l4, self.action_size, name='q')

            self.q_action = tf.argmax(self.q, dimension=1)

            q_summary = []
            avg_q = tf.reduce_mean(self.q, 0)
            for idx in range(self.action_size):
                q_summary.append(tf.summary.histogram('q/%s' % idx, avg_q[idx]))
            self.q_summary = tf.summary.merge(q_summary, 'q_summary')

    def load_pkl(self, pkl_file):
        with open(os.path.join(pkl_file), 'rb') as f:
            return pickle.load(f)

    def train(self, sess):
        print('Training starts..')
        bs = self.batch_size

        for epoch in range(self.num_epochs):
            p_idx = 0
            pkl_action = self.a_list[p_idx]
            pkl_state = self.s_list[p_idx]
            assert pkl_action[:-5] == pkl_state[:-5]
            buff_actions = self.load_pkl(pkl_action)
            buff_states = self.load_pkl(pkl_state)


            for p_idx in len(self.a_list):

            batch_actions =

        return

def main():
    data_path = 'data'
    model = SimpleCNN(task='reach')
    model.set_datapath(data_path)

    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    with tf.Session(config=gpu_config) as sess:
        model.train(sess)

if __name__=='__main__':
    main()