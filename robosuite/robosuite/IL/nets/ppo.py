from mpi4py import MPI
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
import random
import copy
import time
import os


class Agent:
    def __init__(self, env, arg_parser, agent_id):
        self.id = agent_id
        self.name = arg_parser.parse_string('agent_name')
        self.save_name = arg_parser.parse_string('save_name')
        #self.checkpoint_dir = self.save_name+'/checkpoint/'+self.name
        self.checkpoint_dir = 'CHECKPOINT/'+self.name
        self.model_name = self.name.lower()
        # self.action_dim = env.action_space.n
        self.action_dim = env.action_space.shape[0]
        self.state_dim = list(env.observation_space['img'].shape)
        #self.state_dim = env.observation_space.shape[0]
        self.action_bound_min = env.action_space.low
        self.action_bound_max = env.action_space.high

        self.conv1_channels = arg_parser.parse_int('conv1_channels')
        self.conv2_channels = arg_parser.parse_int('conv2_channels')
        self.conv3_channels = arg_parser.parse_int('conv3_channels')
        self.fc1_units = arg_parser.parse_int('fc1_units')
        self.fc2_units = arg_parser.parse_int('fc2_units')

        self.discount_factor = arg_parser.parse_float('discount_factor')
        self.clip_value = arg_parser.parse_float('clip_value')
        self.value_epochs = arg_parser.parse_int('value_epochs')
        self.epochs = arg_parser.parse_int('policy_epochs')
        self.value_lr = arg_parser.parse_float('value_lr')
        self.policy_lr = arg_parser.parse_float('policy_lr')
        self.c1 = 1.0
        self.c2 = 0.0 #0.01

        with tf.variable_scope(self.model_name):
            self.states = tf.placeholder(tf.float32, [None] + self.state_dim, name='State')
            self.actions = tf.placeholder(tf.float32, [None, self.action_dim], name='Action')
            self.gaes = tf.placeholder(tf.float32, [None], name='gaes')
            self.old_std = tf.placeholder(tf.float32, [None, self.action_dim], name='old_std')
            self.old_mean = tf.placeholder(tf.float32, [None, self.action_dim], name='old_mean')
            self.targets = tf.placeholder(tf.float32, [None], name='returns')

            #model 짜기
            self.mean, self.std = self.build_policy_model('policy')
            self.value = self.build_value_model('value')
            self.norm_noise_action = self.mean + tf.random_normal(tf.shape(self.mean))*self.std
            self.sample_noise_action = self.unnormalize_action(self.norm_noise_action)
            self.norm_action = self.mean
            self.sample_action = self.unnormalize_action(self.norm_action)

            #엔트로피 구하기
            self.kl, self.entropy = self.kl_entropy()

            #train 모델 짜기
            self.train_op, self.value_train_op = self.build_op()

            #define sync operator
            self.train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)
            self.sync_vars_ph = []
            self.sync_op = []
            for v in self.train_vars:
                self.sync_vars_ph.append(tf.placeholder(tf.float32, shape=v.get_shape()))
                self.sync_op.append(v.assign(self.sync_vars_ph[-1]))

            if self.id == 0:
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
            else:
                config = tf.ConfigProto(device_count={'GPU': 0})
            self.sess = tf.Session(config=config)
            self.load()


    #define sync operator
    def sync(self):
        if self.id == 0:
            train_vars = self.sess.run(self.train_vars)
        else :
            train_vars = None
        train_vars = MPI.COMM_WORLD.bcast(train_vars, root=0)
        if self.id == 0:
            return
        feed_dict = dict({s_ph: s for s_ph, s in zip(self.sync_vars_ph, train_vars)})
        self.sess.run(self.sync_op, feed_dict=feed_dict)


    def normalize_action(self, a):
        temp_a = 2.0/(self.action_bound_max - self.action_bound_min)
        temp_b = (self.action_bound_max + self.action_bound_min)/(self.action_bound_min - self.action_bound_max)
        temp_a = tf.ones_like(a)*temp_a
        temp_b = tf.ones_like(a)*temp_b
        return temp_a*a + temp_b


    def unnormalize_action(self, a):
        temp_a = (self.action_bound_max - self.action_bound_min)/2.0
        temp_b = (self.action_bound_max + self.action_bound_min)/2.0
        temp_a = tf.ones_like(a)*temp_a
        temp_b = tf.ones_like(a)*temp_b
        return temp_a*a + temp_b
        

    def build_policy_model(self, name='policy'):
        with tf.variable_scope(name):
            model = tf.layers.conv2d(self.states, self.conv1_channels, 8, (4,4), 'valid', 'channels_first', name='conv_0', activation=None,
                                     kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.2))
            model = tf.layers.batch_normalization(model)
            model = tf.nn.relu(model)
            model = tf.layers.conv2d(model, self.conv2_channels, 4, (2, 2), 'valid', 'channels_first', name='conv_1', activation=None,
                                     kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.2))
            model = tf.layers.batch_normalization(model)
            model = tf.nn.relu(model)
            model = tf.layers.conv2d(model, self.conv3_channels, 3, (1, 1), 'valid', 'channels_first', name='conv_2', activation=None,
                                     kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.2))
            model = tf.layers.batch_normalization(model)
            model = tf.nn.relu(model)
            model = tf.layers.flatten(model, name='flat_0', data_format='channels_first')

            model = tf.layers.dense(model, self.fc1_units, name='dense_0', activation=None,
                                    kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            model = tf.nn.relu(model)
            model = tf.layers.dense(model, self.fc2_units, name='dense_1', activation=None,
                                    kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            model = tf.nn.relu(model)

            mean = tf.layers.dense(model, self.action_dim, activation=tf.tanh,
                                   kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02), name="mean")

            # return mean
            logits_std = tf.get_variable("logits_std", shape=(self.action_dim),
                                         initializer=tf.random_normal_initializer(mean=-1.0,
                                                                                  stddev=0.02))  # 0.1정도로 initialize
            std = tf.ones_like(mean) * tf.nn.softplus(logits_std)
            return mean, std

            '''
            model = tf.layers.dense(self.states, self.hidden1_units, name='dense_0', activation=None, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            model = tf.layers.batch_normalization(model)
            model = tf.nn.relu(model)
            model = tf.layers.dense(model, self.hidden2_units, name='dense_1', activation=None, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            model = tf.layers.batch_normalization(model)
            model = tf.nn.relu(model)
            
            model = tf.layers.dense(model, self.hidden2_units, name='dense_2', activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            mean = tf.layers.dense(model, self.action_dim, activation=tf.tanh, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02), name="mean")

            #return mean
            logits_std = tf.get_variable("logits_std",shape=(self.action_dim),initializer=tf.random_normal_initializer(mean=-1.0,stddev=0.02)) # 0.1정도로 initialize
            std = tf.ones_like(mean)*tf.nn.softplus(logits_std)
            return mean, std
            '''

    def build_value_model(self, name='value'):
        with tf.variable_scope(name):
            model = tf.layers.conv2d(self.states, self.conv1_channels, 8, (4, 4), 'valid', 'channels_first', name='conv_0',
                                     activation=None,
                                     kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.2))
            model = tf.layers.batch_normalization(model)
            model = tf.nn.relu(model)
            model = tf.layers.conv2d(model, self.conv2_channels, 4, (2, 2), 'valid', 'channels_first', name='conv_1',
                                     activation=None,
                                     kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.2))
            model = tf.layers.batch_normalization(model)
            model = tf.nn.relu(model)
            model = tf.layers.conv2d(model, self.conv3_channels, 3, (1, 1), 'valid', 'channels_first', name='conv_2',
                                     activation=None,
                                     kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.2))
            model = tf.layers.batch_normalization(model)
            model = tf.nn.relu(model)
            model = tf.layers.flatten(model, name='flat_0', data_format='channels_first')

            model = tf.layers.dense(model, self.fc1_units, name='dense_0', activation=None,
                                    kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            model = tf.nn.relu(model)
            model = tf.layers.dense(model, self.fc2_units, name='dense_1', activation=None,
                                    kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            model = tf.nn.relu(model)
            model = tf.layers.dense(model, 1, name='dense_2', activation=None,
                                          kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            value = tf.reshape(model, [-1])
            return value
            '''
            model = tf.layers.dense(self.states, self.hidden1_units, name='dense_0', activation=None, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            model = tf.layers.batch_normalization(model)
            model = tf.nn.relu(model)
            model = tf.layers.dense(model, self.hidden2_units, name='dense_1', activation=None, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            model = tf.layers.batch_normalization(model)
            model = tf.nn.relu(model)
            model = tf.layers.dense(model, self.hidden2_units, name='dense_2', activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            model = tf.layers.dense(model, 1, activation=None, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            model = tf.reshape(model, [-1])
            return model
            '''

    def build_op(self):
        actions = self.normalize_action(self.actions)
        log_policy = tf.reduce_sum(-0.5*tf.square((actions - self.mean)/self.std) - tf.log(self.std) - 0.5*np.log(2*np.pi), axis=1)
        log_policy_old = tf.reduce_sum(-0.5*tf.square((actions - self.old_mean)/self.old_std) - tf.log(self.old_std) - 0.5*np.log(2*np.pi), axis=1)
        ratios = tf.exp(log_policy - log_policy_old)
        clipped_ratios = tf.clip_by_value(ratios, clip_value_min=1 - self.clip_value, clip_value_max=1 + self.clip_value)
        loss_clip = tf.minimum(tf.multiply(self.gaes, ratios), tf.multiply(self.gaes, clipped_ratios))
        self.loss_policy = -self.c1*tf.reduce_mean(loss_clip) - self.c2*self.entropy

        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name+'/policy')
        optimizer = tf.train.AdamOptimizer(learning_rate=self.policy_lr)
        grads = tf.gradients(self.loss_policy, train_vars)
        train_op = optimizer.apply_gradients(zip(grads, train_vars))

        loss_value = 0.5*tf.square(self.targets - self.value)
        self.loss_value = tf.reduce_mean(loss_value)

        train_vars2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name+'/value')
        optimizer2 = tf.train.AdamOptimizer(learning_rate=self.value_lr)
        grads2 = tf.gradients(self.loss_value, train_vars2)
        train_op2 = optimizer2.apply_gradients(zip(grads2, train_vars2))
            
        return train_op, train_op2


    def kl_entropy(self):
        mean, std = self.mean, self.std
        old_mean, old_std = self.old_mean, self.old_std
        log_std_old = tf.log(old_std)
        log_std_new = tf.log(std)
        frac_std_old_new = old_std/std
        kl = tf.reduce_mean(log_std_new - log_std_old + 0.5*tf.square(frac_std_old_new) + 0.5*tf.square((mean - old_mean)/std)- 0.5)
        entropy = tf.reduce_mean(log_std_new + 0.5 + 0.5*np.log(2*np.pi))
        return kl, entropy


    def get_action(self, state, is_train):
        if is_train:
            [action] = self.sess.run(self.sample_noise_action, feed_dict={self.states:[state]})
        else:
            [action] = self.sess.run(self.sample_action, feed_dict={self.states:[state]})
        action = np.clip(action, self.action_bound_min, self.action_bound_max)
        [value] = self.sess.run(self.value, feed_dict={self.states: [state]})
        return action, value


    def train(self, inp, batch_size=64, only_value=False): # TRAIN POLICY
        states = inp[0]
        actions = inp[1]
        gaes = inp[2]
        targets = inp[3]
        old_means, old_stds = self.sess.run([self.mean, self.std],feed_dict={self.states: states})

        num_batches = states.shape[0] // batch_size
        if not only_value:
            for _ in range(self.epochs):
                states, actions, gaes, targets, old_means, old_stds = shuffle(states, actions, gaes, targets, old_means, old_stds, random_state=0)
                for j in range(num_batches): 
                    start = j * batch_size
                    end = start + batch_size
                    self.sess.run(self.train_op, \
                        feed_dict={self.states:states[start:end], \
                        self.actions:actions[start:end], \
                        self.gaes:gaes[start:end], \
                        self.old_mean:old_means[start:end], \
                        self.old_std:old_stds[start:end], \
                        self.targets:targets[start:end]})

        for _ in range(self.value_epochs):
            states, actions, gaes, targets, old_means, old_stds = shuffle(states, actions, gaes, targets, old_means, old_stds, random_state=0)
            for j in range(num_batches): 
                start = j * batch_size
                end = start + batch_size
                self.sess.run(self.value_train_op, \
                    feed_dict={self.states:states[start:end], \
                    self.actions:actions[start:end], \
                    self.gaes:gaes[start:end], \
                    self.old_mean:old_means[start:end], \
                    self.old_std:old_stds[start:end], \
                    self.targets:targets[start:end]})

        loss_p, loss_v, kl, entropy = self.sess.run([self.loss_policy, self.loss_value, self.kl, self.entropy], \
                    feed_dict={self.states:states, \
                    self.actions:actions, \
                    self.gaes:gaes, \
                    self.old_mean:old_means, \
                    self.old_std:old_stds, \
                    self.targets:targets})

        return loss_p, loss_v, kl, entropy


    def get_entropy(self):
        temp = np.ones((1, self.state_dim))
        entropy = self.sess.run(self.entropy, feed_dict={self.states:temp})
        return entropy


    def get_gaes(self, rewards, next_values, values, gae_coef = 0.98):
        deltas = [r_t + self.discount_factor * v_next - v for r_t, v_next, v in zip(rewards, next_values, values)]
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(gaes) - 1)):  # is T-1, where T is time step which run policy
            gaes[t] = gaes[t] + self.discount_factor * gae_coef * gaes[t + 1]
        return gaes
        #return deltas


    def get_gaes_targets(self, trajectories, r_ts):
        targets = []
        gaes = []
        score = 0
        for i, t in enumerate(trajectories):
            rewards = r_ts[i]
            gae = self.get_gaes(rewards, t[3], t[2])
            target = np.zeros_like(rewards)
            ret = 0
            for tt in reversed(range(len(rewards))):
                ret = rewards[tt] + self.discount_factor*ret
                target[tt] = ret
            gae = np.array(gae).astype(dtype=np.float32)
            target = np.array(target).astype(dtype=np.float32)
            gaes.append(gae)
            targets.append(target)
            score += np.sum(rewards)
        gaes = np.concatenate(gaes)
        targets = np.concatenate(targets)
        gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-6)
        return gaes, targets


    def save(self):
        self.saver.save(self.sess, self.checkpoint_dir+'/model.ckpt')
        print('[{}] save success!'.format(self.name))


    def load(self):
        self.saver = tf.train.Saver(var_list= tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name))

        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print('[{}] success to load model!'.format(self.name))
        else:
            self.sess.run(tf.global_variables_initializer())
            print('[{}] fail to load model...'.format(self.name))
        
