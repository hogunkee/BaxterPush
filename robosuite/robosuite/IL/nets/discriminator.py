from sklearn.utils import shuffle
from collections import deque
import tensorflow as tf
import numpy as np
import pickle
import copy
import os

class Discriminator:
    def __init__(self, env, arg_parser, expert_states, expert_actions, agent_id):
        self.id = agent_id
        self.name = arg_parser.parse_string('disc_name')
        self.save_name = arg_parser.parse_string('save_name')
        self.checkpoint_dir = 'CHECKPOINT/'+self.name
        self.model_name = self.name.lower()

        self.state_dim = list(env.observation_space['img'].shape)
        self.action_dim = env.action_space.shape[0]

        self.conv1_channels = arg_parser.parse_int('conv1_channels')
        self.conv2_channels = arg_parser.parse_int('conv2_channels')
        # self.conv3_channels = arg_parser.parse_int('conv3_channels')
        self.sfc1_units = arg_parser.parse_int('sfc1_units')
        # self.sfc2_units = arg_parser.parse_int('sfc2_units')
        self.fc1_units = arg_parser.parse_int('fc1_units')
        # self.fc2_units = arg_parser.parse_int('fc2_units')

        self.lr = arg_parser.parse_float('disc_lr')
        self.weight_gp = arg_parser.parse_float('weight_gp')
        self.train_epochs = arg_parser.parse_int('disc_epochs')

        self.expert_states = expert_states
        self.expert_actions = expert_actions

        self.replay_buffer_s = deque(maxlen=10000)
        self.replay_buffer_a = deque(maxlen=10000)

        with tf.variable_scope(self.model_name):
            self.expert_s = tf.placeholder(dtype=tf.float32, shape=[None] + self.state_dim, name='expert_states')
            self.expert_a = tf.placeholder(dtype=tf.float32, shape=[None, self.action_dim], name='expert_actions')
            # add noise for stabilise training
            expert_a = tf.clip_by_value(self.expert_a + tf.random_normal(tf.shape(self.expert_a), mean=0.0, stddev=0.02, dtype=tf.float32), -1.0, 1.0)
            #expert_s_a = tf.concat([self.expert_s, expert_a], -1)
            #expert_s_a = tf.concat([self.expert_s, self.expert_a], -1)

            self.agent_s = tf.placeholder(dtype=tf.float32, shape=[None] + self.state_dim, name='states')
            self.agent_a = tf.placeholder(dtype=tf.float32, shape=[None, self.action_dim], name='actions')
            #agent_s_a = tf.concat([self.agent_s, self.agent_a], -1)

            expert_result = self.build_model(self.expert_s, expert_a)
            agent_result = self.build_model(self.agent_s, self.agent_a, True)

            #discriminator가 잘 평가했는지 측정
            self.agent_acc = tf.reduce_mean(tf.to_float(agent_result > 0.5))
            self.expert_acc = tf.reduce_mean(tf.to_float(expert_result < 0.5))
            
            #loss 계산
            self.agent_loss = tf.reduce_mean(tf.log(agent_result+1e-8))
            self.expert_loss = tf.reduce_mean(tf.log(1 - expert_result+1e-8))
            self.each_agent_loss = tf.reduce_mean(tf.log(agent_result+1e-8), axis=1)
            total_result = tf.concat([expert_result, agent_result], 0)
            self.entropy = -tf.reduce_mean(total_result*tf.log(tf.clip_by_value(total_result, 1e-10, 1 - 1e-10)))
            #################### gradient punishment 추가!! ############################
            eps_s = tf.random_uniform(tf.shape(self.agent_s), 0.0, 1.0)
            eps_a = tf.random_uniform(tf.shape(self.agent_a), 0.0, 1.0)
            s_eps = eps_s*self.expert_s + (1-eps_s)*self.agent_s
            a_eps = eps_a*self.expert_a + (1-eps_a)*self.agent_a
            r_eps = self.build_model(s_eps, a_eps, reuse=True)
            # d_input = tf.concat([s_eps,a_eps], axis=-1)
            # r_eps = self.build_model(d_input, True)
            # gradients = tf.gradients(r_eps, d_input)[0]
            gradients = tf.gradients(r_eps, [s_eps, a_eps])
            gradients = tf.concat([tf.reshape(gradients[0], [-1, 3*84*84]), gradients[1]], axis=-1)
            grad_l2 = tf.sqrt(tf.reduce_sum(tf.square(gradients),axis=-1))
            self.gp_loss = tf.reduce_mean(tf.square(grad_l2 - 1.0))
            #################### gradient punishment 추가!! ############################
            self.loss = -(self.agent_loss + self.expert_loss) + self.weight_gp*self.gp_loss

            train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            self.train_op = optimizer.minimize(self.loss, var_list=train_vars)

            self.reward = tf.reduce_sum(-tf.log(agent_result+ 1e-8), axis=1)

            if self.id == 0:
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
            else:
                config = tf.ConfigProto(device_count={'GPU': 0})
            self.sess = tf.Session(config=config)
            self.load()


    def build_model(self, state, action, reuse=False):
        with tf.variable_scope('networks', reuse=reuse):
            model = tf.layers.conv2d(state, self.conv1_channels, 8, (4, 4), 'valid', 'channels_first',
                                     name='conv_0', activation=None,
                                     kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.2))
            model = tf.layers.batch_normalization(model)
            model = tf.nn.relu(model)
            # model = tf.layers.conv2d(model, self.conv2_channels, 4, (2, 2), 'valid', 'channels_first', name='conv_1',
            model = tf.layers.conv2d(model, self.conv2_channels, 5, (3, 3), 'valid', 'channels_first', name='conv_1',
                                     activation=None,
                                     kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.2))
            model = tf.layers.batch_normalization(model)
            model = tf.nn.relu(model)
            # model = tf.layers.conv2d(model, self.conv3_channels, 3, (1, 1), 'valid', 'channels_first', name='conv_2',
            #                          activation=None,
            #                          kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.2))
            # model = tf.layers.batch_normalization(model)
            # model = tf.nn.relu(model)
            model = tf.layers.flatten(model, name='flat_0', data_format='channels_first')

            model = tf.layers.dense(model, self.sfc1_units, name='dense_0', activation=None,
                                    kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            model = tf.nn.relu(model)
            # model = tf.layers.dense(model, self.sfc2_units, name='dense_1', activation=None,
            #                         kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            # model = tf.nn.relu(model)

            state_action = tf.concat([model, action], axis=-1)
            #layer = tf.layers.dense(inputs=input, units=64, activation=tf.nn.tanh, name='layer1')
            layer = tf.layers.dense(inputs=state_action, units=self.fc1_units, activation=tf.nn.tanh, name='layer1')
            # layer = tf.layers.dense(inputs=layer, units=self.fc2_units, activation=tf.nn.tanh, name='layer2')
            disc = tf.layers.dense(inputs=layer, units=1, activation=tf.sigmoid, name='disc')
            return disc


    '''
    def train(self, s, a, states, actions, batch_size=64):
        data_size = min(s.shape[0], self.expert_states.shape[0])
        num_batches = max(data_size // batch_size, 1)
        batch_size = data_size // num_batches

        expert_states = self.expert_states
        expert_actions = self.expert_actions

        for _ in range(self.train_epochs):
            states, actions = shuffle(states, actions, random_state=0)
            expert_states, expert_actions = shuffle(expert_states, expert_actions, random_state=0)

            a_losses = []
            e_losses = []
            gp_losses = []

            for j in range(num_batches): 
                start = j * batch_size
                end = (j + 1) * batch_size
                _, a_l,e_l,g_l = self.sess.run([self.train_op, self.agent_loss, self.expert_loss, self.gp_loss], \
                                    feed_dict={self.expert_s:expert_states[start:end], self.expert_a:expert_actions[start:end], \
                                    self.agent_s:states[start:end], self.agent_a:actions[start:end]})
                a_losses.append(a_l)
                e_losses.append(e_l)
                gp_losses.append(g_l)

        a_acc, e_acc, entropy = self.sess.run([self.agent_acc, self.expert_acc, self.entropy], \
                            feed_dict={self.expert_s:self.expert_states, self.expert_a:self.expert_actions, \
                            self.agent_s:s, self.agent_a:a})

        return a_acc, e_acc, entropy, np.mean(a_losses), np.mean(e_losses), np.mean(gp_losses)
    '''
    def train(self, s, a, batch_size=64):
        data_size = min(s.shape[0], self.expert_states.shape[0])
        num_batches = max(data_size // batch_size, 1)
        batch_size = data_size // num_batches

        expert_states = self.expert_states
        expert_actions = self.expert_actions

        self.replay_buffer_s += deque(s)
        self.replay_buffer_a += deque(a)
        states = self.replay_buffer_s
        actions = self.replay_buffer_a

        for _ in range(self.train_epochs):
            states, actions = shuffle(states, actions, random_state=0)
            expert_states, expert_actions = shuffle(expert_states, expert_actions, random_state=0)

            a_losses = []
            e_losses = []
            gp_losses = []

            a_acces = []
            e_acces = []
            entropies = []

            for j in range(num_batches): 
                start = j * batch_size
                end = (j + 1) * batch_size
                _, a_l, e_l, g_l, a_a, e_a, ent = self.sess.run([self.train_op, self.agent_loss, self.expert_loss, self.gp_loss] \
                                                                + [self.agent_acc, self.expert_acc, self.entropy], \
                                                 feed_dict={self.expert_s: expert_states[start:end],
                                                            self.expert_a: expert_actions[start:end], \
                                                            self.agent_s: states[start:end],
                                                            self.agent_a: actions[start:end]})
                # _, a_l,e_l,g_l = self.sess.run([self.train_op, self.agent_loss, self.expert_loss, self.gp_loss], \
                #                     feed_dict={self.expert_s:expert_states[start:end], self.expert_a:expert_actions[start:end], \
                #                     self.agent_s:states[start:end], self.agent_a:actions[start:end]})
                a_losses.append(a_l)
                e_losses.append(e_l)
                gp_losses.append(g_l)

                a_acces.append(a_a)
                e_acces.append(e_a)
                entropies.append(ent)

        a_acc = np.mean(a_acces)
        e_acc = np.mean(e_acces)
        entropy = np.mean(entropies)
        # a_acc, e_acc, entropy = self.sess.run([self.agent_acc, self.expert_acc, self.entropy], \
        #                     feed_dict={self.expert_s:self.expert_states, self.expert_a:self.expert_actions, \
        #                     self.agent_s:states, self.agent_a:actions})

        return a_acc, e_acc, entropy, np.mean(a_losses), np.mean(e_losses), np.mean(gp_losses)


    def get_measure_distance(self, states, actions):
        #e_loss, a_loss = self.sess.run([self.expert_loss, self.each_agent_loss], \
        e_loss, a_loss = self.sess.run([self.expert_loss, self.agent_loss], \
                        feed_dict={self.expert_s:self.expert_states, \
                                self.expert_a:self.expert_actions, \
                                self.agent_s:states, \
                                self.agent_a:actions})
        return e_loss + a_loss


    def get_acc(self, states, actions):
        a_acc, e_acc = self.sess.run([self.agent_acc, self.expert_acc], \
                            feed_dict={self.expert_s:self.expert_states, self.expert_a:self.expert_actions, \
                            self.agent_s:states, self.agent_a:actions})
        return a_acc, e_acc


    def get_reward(self, states, actions):
        return self.sess.run(self.reward, feed_dict={self.agent_s:states, self.agent_a:actions})


    def save(self):
        self.saver.save(self.sess, self.checkpoint_dir+'/disc_model.ckpt')
        #replay_buffer save
        with open("{}/replay_buffer.pkl".format(self.checkpoint_dir), 'wb') as f:
            pickle.dump([self.replay_buffer_s, self.replay_buffer_a], f)
        ######################
        print('[{}] save success!'.format(self.name))


    def load(self):
        self.saver = tf.train.Saver(var_list= tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name))

        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            #replay_buffer load
            with open("{}/replay_buffer.pkl".format(self.checkpoint_dir), 'rb') as f:
                try:
                    self.replay_buffer_s, self.replay_buffer_a = pickle.load(f)
                except EOFError:
                    self.replay_buffer_s, self.replay_buffer_a = [], []
            ######################
            print('[{}] success to load model!'.format(self.name))
        else:
            self.sess.run(tf.global_variables_initializer())
            print('[{}] fail to load model...'.format(self.name))
