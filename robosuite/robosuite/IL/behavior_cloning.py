import datetime
import numpy as np
import os
import sys
import pickle

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '../..'))
sys.path.append(os.path.join(FILE_PATH, '..', 'scripts'))
from demo_baxter_rl_pushing import *

from functools import reduce
from ops import *
from tensorboardX import SummaryWriter

FILE_PATH = os.path.dirname(os.path.abspath(__file__))

class SimpleCNN():
    def __init__(self, task, action_size, cnn_format='NHWC', model_name=None):
        self.task = task
        self.action_size = action_size

        self.cnn_format = cnn_format
        self.screen_height = 64
        self.screen_width = 64
        self.screen_channel = 4

        self.data_path = None
        self.num_epochs = 100
        self.batch_size = 128
        self.lr = 1e-3
        self.loss_type = 'l2' # 'l2' or 'ce'
        self.test_freq = 20
        self.num_test_ep = 1
        self.env = None

        self.dueling = False
        self.now = datetime.datetime.now()
        if model_name is None:
            self.model_name = self.task + '_' + self.now.strftime("%m%d_%H%M%S")
        else:
            self.model_name = model_name
        self.checkpoint_dir = os.path.join(FILE_PATH, 'bc_train_log/', 'checkpoint/', self.model_name)

        self.build_net()
        self.saver = tf.train.Saver(max_to_keep=10)
        self.max_accur = 0.0


    def set_datapath(self, data_path):
        file_list = os.listdir(data_path)
        pkl_list = [f for f in file_list if 'pkl' in f and self.task in f]
        if len(pkl_list)==0:
            print('No pickle files exist. Wrong data path!!')
            return
        self.pkl_list = sorted([os.path.join(data_path, p) for p in pkl_list])
        self.a_list = sorted([p for p in pkl_list if self.task + '_a_' in p])
        self.s_list = sorted([p for p in pkl_list if self.task + '_s_' in p])
        assert len(self.a_list) == len(self.s_list)
        self.data_path = data_path

    def build_net(self):
        self.w = {}
        self.t_w = {}

        #initializer = tf.contrib.layers.xavier_initializer()
        initializer = tf.truncated_normal_initializer(0, 0.1) #0.02)
        activation_fn = tf.nn.relu

        # training network
        self.a_true = tf.placeholder('int64', [None], name='a_t')
        with tf.variable_scope('prediction'):
            if self.cnn_format == 'NHWC':
                self.s_t = tf.placeholder('float32', [None, 2, self.screen_height, self.screen_width, self.screen_channel], name='s_t')
            else:
                self.s_t = tf.placeholder('float32', [None, 2, self.screen_channel, self.screen_height, self.screen_width], name='s_t')

            self.s_t_0 = self.s_t[:, 0, :, :, :]
            self.s_t_1 = self.s_t[:, 1, :, :, :]
            self.s_t_concat = tf.concat([self.s_t_0, self.s_t_1], axis=-1)

            self.l1, self.w['l1_w'], self.w['l1_b'] = conv2d(self.s_t_concat,
                32, [3, 3], [2, 2], initializer, activation_fn, self.cnn_format, padding='SAME', name='l1')
            self.l2, self.w['l2_w'], self.w['l2_b'] = conv2d(self.l1,
                64, [3, 3], [2, 2], initializer, activation_fn, self.cnn_format, padding='SAME', name='l2')
            self.l3, self.w['l3_w'], self.w['l3_b'] = conv2d(self.l2,
                128, [3, 3], [2, 2], initializer, activation_fn, self.cnn_format, padding='SAME', name='l3')
            self.l4, self.w['l4_w'], self.w['l4_b'] = conv2d(self.l3,
                256, [3, 3], [2, 2], initializer, activation_fn, self.cnn_format, padding='SAME', name='l4')

            self.l4_pool = tf.reduce_mean(self.l4, [1, 2])


            self.l5, self.w['l5_w'], self.w['l5_b'] = linear(self.l4_pool, 128, activation_fn=activation_fn, name='l5')
            self.l6, self.w['l6_w'], self.w['l6_b'] = linear(self.l5, 128, activation_fn=activation_fn, name='l6')
            self.q, self.w['q_w'], self.w['q_b'] = linear(self.l6, self.action_size, name='q')

            self.q_action = tf.argmax(self.q, dimension=1)

            if self.loss_type=='ce':
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.q, labels=tf.one_hot(self.a_true, depth=self.action_size))
            elif self.loss_type=='l2':
                cross_entropy = tf.nn.l2_loss(self.q - tf.one_hot(self.a_true, depth=self.action_size))

            self.cost = tf.reduce_mean(cross_entropy)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.cost)

            self.correct_prediction = tf.equal(self.q_action, self.a_true)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))


    def load_pkl(self, pkl_file):
        with open(os.path.join(self.data_path, pkl_file), 'rb') as f:
            return pickle.load(f)

    def load_model(self, sess):
        print(" [*] Loading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            fname = os.path.join(self.checkpoint_dir, ckpt_name)
            self.saver.restore(sess, fname)
            print(" [*] Load SUCCESS: %s" % fname)
            return True
        else:
            print(" [!] Load FAILED: %s" % self.checkpoint_dir)
            return False

    def test_agent(self, sess):
        if self.env is None:
            return
        success_log = []
        for n in range(self.num_test_ep):
            obs = self.env.reset()
            done = False
            cumulative_reward = 0.0
            step_count = 0

            while not done:
                step_count += 1
                action = sess.run(self.q_action, feed_dict={self.s_t: [obs]})
                obs, reward, done, _ = self.env.step(action)
                print('action: %d / reward: %.2f' % (action, reward))
                # print(step_count, 'steps \t action: ', action, '\t reward: ', reward)
                cumulative_reward += reward

            if reward >= 100:
                success_log.append(1)
            else:
                success_log.append(0)
            print(step_count, cumulative_reward)

        print(success_log)
        print('success rate?:', np.mean(success_log))
        return

    def set_env(self, env):
        self.env = env

    def train(self, sess):
        writer = SummaryWriter(os.path.join(FILE_PATH, 'bc_train_log/', 'tensorboard', self.task + '_' + self.now.strftime("%m%d_%H%M%S")))
        sess.run(tf.global_variables_initializer())

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        print('Training starts..')
        bs = self.batch_size
        for epoch in range(self.num_epochs):
            if epoch==40:
                self.lr /= 10.0
            elif epoch==70:
                self.lr /= 10.0

            epoch_cost = []
            epoch_accur = []
            for p_idx in np.random.permutation(len(self.a_list)):
                pkl_action = self.a_list[p_idx]
                pkl_state = self.s_list[p_idx]
                assert pkl_action[-5:] == pkl_state[-5:]
                buff_actions = self.load_pkl(pkl_action)[1:]
                buff_states = self.load_pkl(pkl_state)[:-1]
                assert len(buff_actions) == len(buff_states)

                shuffler = np.random.permutation(len(buff_actions))
                buff_actions = buff_actions[shuffler]
                buff_states = buff_states[shuffler]
                buff_states = np.clip(buff_states, 0.0, 5.0)

                for i in range(len(buff_actions)//bs):
                    batch_actions = buff_actions[bs * i:bs * (i + 1)]
                    batch_states = buff_states[bs * i:bs * (i + 1)]
                    _, cost, accuracy = sess.run([self.optimizer, self.cost, self.accuracy], \
                                                   feed_dict={self.s_t: batch_states, self.a_true: batch_actions})
                    epoch_cost.append(cost)
                    epoch_accur.append(accuracy)

            writer.add_scalar('train-%s/mean_cost'%self.task, np.mean(epoch_cost))
            writer.add_scalar('train-%s/mean_accuracy'%self.task, np.mean(epoch_accur))
            print('[Epoch %d] cost: %.3f\taccur: %.3f' %(epoch, np.mean(epoch_cost), np.mean(epoch_accur)))

            # save the model parameters
            if np.mean(epoch_accur) > 0.90 and np.mean(epoch_accur) > self.max_accur:
                self.saver.save(sess, os.path.join(self.checkpoint_dir, 'model'), global_step=epoch)
                self.max_accur = np.mean(epoch_accur)
            # performance evaluation
            if (epoch+1) % self.test_freq == 0:
                self.test_agent(sess)

        print('Training done!')
        return


def main():
    task = 'push' # 'reach' / 'push'
    action_type = '3D' # '2D' / '3D'

    render = True
    screen_width = 64
    screen_height = 64
    rgbd = True

    env = robosuite.make(
        "BaxterPush",
        bin_type='table',
        object_type='cube',
        ignore_done=True,
        has_renderer=True,
        camera_name="eye_on_right_wrist",
        gripper_visualization=False,
        use_camera_obs=False,
        use_object_obs=False,
        camera_depth=True,
        num_objects=2,
        control_freq=100,
        camera_width=screen_width,
        camera_height=screen_height,
        crop=None
    )
    env = IKWrapper(env)
    env = BaxterEnv(env, task=task, render=render, using_feature=False, rgbd=rgbd, action_type=action_type)

    data_path = 'data'
    model = SimpleCNN(task=task, action_size=env.action_size)
    model.set_datapath(data_path)
    model.set_env(env)

    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    with tf.Session(config=gpu_config) as sess:
        model.train(sess)


if __name__=='__main__':
    main()