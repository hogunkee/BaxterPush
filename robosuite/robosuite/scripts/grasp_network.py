import numpy as np
import math
import json
import tensorflow as tf
import tensorflow.contrib.framework as tcf
from tensorflow.python.keras import backend
from tensorflow.python.keras import initializers
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import regularizers
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="6"
EPS = 1e-8
L2_WEIGHT_DECAY = 1e-4
BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5
LOG_SIG_CAP_MAX = 2
LOG_SIG_CAP_MIN = -20

def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()
    
class VPPNET(object):
    def __init__(self, config, model_dir=None, is_training=True):
        self._config = config
        self._is_training = is_training
        self._momentum_rate = 0.9
        self._max_global_grad_norm = 100000000000
        self._train_l2_regularizer = 0.0005
        
        data_dir = "/home/rllab/robosuite-gqcnn/robosuite/robosuite/scripts/metrics"#self._config["data_dir"]
        _im_depth_mean_filename = os.path.join(data_dir, 'vt_im_depth_mean.npy')
        _im_depth_std_filename = os.path.join(data_dir, 'vt_im_depth_std.npy')
        self._im_depth_mean = np.load(_im_depth_mean_filename)
        self._im_depth_std = np.load(_im_depth_std_filename)
        
        self._model_type = self._config["model_type"]
        self._num_mixtures = self._config["num_mixtures"]
        self._batch_size = self._config["batch_size"]
        self._im_length = self._config["image_length"]
        self._im_width = self._config["image_width"]
        self._im_channels = 1
        self._pose_dim = self._config["pose_dim"]
        self._drop_rate = self._config["drop_rate"]
        
        self._weights = {}
        self._optimize_base_layers = True
        self.load_pretrained_model(model_dir=model_dir)
        
    def load_pretrained_model(self, model_dir=None):
        self.g = tf.Graph()
        self._build_graph()
        self._init_session()
        if model_dir is not None:
            self.load_model(model_dir)
            
    def _build_graph(self):
        with self.g.as_default():
            if self._model_type == 'regression' or self._model_type == 'gmm':
                self.input_im_node = tf.placeholder(tf.float32, shape=[None, self._im_length, self._im_width, self._im_channels])
                self.input_pos_node = tf.placeholder(tf.float32, shape=[None, self._pose_dim])
            else:
                print("Invalid Model Type.")
            self.drop_rate_node = tf.placeholder(tf.float32, shape=())    
            self.learning_rate_node = tf.placeholder_with_default(tf.constant(0.01), ())
            self.is_training_node = tf.placeholder(tf.bool)
            
            if self._model_type == 'regression':
                self._output_tensor = self.vpp(self.input_im_node, self.drop_rate_node, self.is_training_node, batch_size=None)
                self._net_output = self._output_tensor
            elif self._model_type == 'gmm':
                self._log_ws, self._mus, self._log_sigs = self.vpp_mixture(self.input_im_node, self.drop_rate_node, self.is_training_node, batch_size=None)
                self._net_output = [self._log_ws, tf.nn.tanh(self._mus), self._mus, self._log_sigs]
                
            if self._is_training:
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                
                if self._model_type == 'regression':
                    self.unregularized_loss = tf.reduce_mean((self._output_tensor - self.input_pos_node) ** 2)
                elif self._model_type == 'gmm':
                    log_p = self.create_log_gaussian(self._mus, self._log_sigs, self.input_pos_node[:, None, :])
                    log_p = tf.reduce_logsumexp(log_p + self._log_ws, axis=1)
                    log_p -= tf.reduce_logsumexp(self._log_ws, axis=1)  # N
                    self.log_p = log_p
                    self.log_p -= self._squash_correction(self.input_pos_node)
                    self.unregularized_loss = -tf.reduce_mean(self.log_p)
                    self._reg = 0.001
                    self.unregularized_loss += self._reg * 0.5 * tf.reduce_mean(self._log_sigs ** 2)
                else:
                    print("Invalid Model Type.")
                self.loss = self.unregularized_loss
                
                t_vars = tf.trainable_variables()
                if self._model_type != 'gmm':
                    self.regularizers = tf.nn.l2_loss(t_vars[0])
                    for var in t_vars[1:]:
                        self.regularizers += tf.nn.l2_loss(var)
                    self.loss += self._train_l2_regularizer * self.regularizers
                
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate_node)
                gradients, variables = zip(*self.optimizer.compute_gradients(self.loss, var_list=t_vars))
                gradients, global_grad_norm = tf.clip_by_global_norm(gradients, self._max_global_grad_norm)
                self.train_op = self.optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)
            ################################ REGRESSION ORIIGINAL #####################
            self.init = tf.global_variables_initializer()
            t_vars = tf.trainable_variables()
            self.assign_ops = {}
            for var in t_vars:
                pshape = var.get_shape()
                pl = tf.placeholder(tf.float32, pshape, var.name[:-2]+'_placeholder')
                assign_op = var.assign(pl)
                self.assign_ops[var] = (assign_op, pl)
                
    def _squash_correction(self, actions):
        return tf.reduce_sum(tf.log(1 - tf.tanh(actions) ** 2 + EPS), axis=1)
    
    def _init_session(self):
        config = tf.ConfigProto()
        self.sess = tf.Session(config=config, graph=self.g)
        self.sess.run(self.init)
    
    def close_session(self):
        self.sess.close()
    
    def predict(self, image_arr, is_training=False):
        norm_sub_im_arr = (image_arr - self._im_depth_mean) / self._im_depth_std
        output = self.sess.run(self._net_output, 
                               feed_dict={self.input_im_node: norm_sub_im_arr, self.drop_rate_node: 0.0,
                                          self.is_training_node: is_training})
        return output
    
    def get_model_params(self):
        model_names = []
        model_params = []
        model_shapes = []
        with self.g.as_default():
            t_vars = tf.trainable_variables()
            for var in t_vars:
                #if var.name.startswith('conv_vae'):
                param_name = var.name
                p = self.sess.run(var)
                model_names.append(param_name)
                params = np.round(p*10000).astype(np.int).tolist()
                model_params.append(params)
                model_shapes.append(p.shape)
        return model_params, model_shapes, model_names
 
    def get_random_model_params(self, stdev=0.5):
        # get random params.
        _, mshape, _ = self.get_model_params()
        rparam = []
        for s in mshape:
            #rparam.append(np.random.randn(*s)*stdev)
            rparam.append(np.random.standard_cauchy(s)*stdev) # spice things up
        return rparam
                
    def set_model_params(self, params):
        with self.g.as_default():
            t_vars = tf.trainable_variables()
            idx = 0
            for var in t_vars:
                #if var.name.startswith('conv_vae'):
                pshape = tuple(var.get_shape().as_list())
                p = np.array(params[idx])
                assert pshape == p.shape, "inconsistent shape"
                assign_op, pl = self.assign_ops[var]
                self.sess.run(assign_op, feed_dict={pl.name: p/10000.})
                idx += 1

    def load_json(self, jsonfile='gqcnn.json'):
        with open(jsonfile, 'r') as f:
            params = json.load(f)
        self.set_model_params(params)
  
    def save_json(self, jsonfile='gqcnn.json'):
        model_params, model_shapes, model_names = self.get_model_params()
        qparams = []
        for p in model_params:
            qparams.append(p)
        with open(jsonfile, 'wt') as outfile:
            json.dump(qparams, outfile, sort_keys=True, indent=0, separators=(',', ': '))
  
    def set_random_params(self, stdev=0.5):
        rparam = self.get_random_model_params(stdev)
        self.set_model_params(rparam)
  
    def save_model(self, model_save_path):
        sess = self.sess
        with self.g.as_default():
            saver = tf.train.Saver(tf.global_variables())
            saver.save(sess, os.path.join(model_save_path, 'model.ckpt'))

    def load_model(self, model_save_path):
        sess = self.sess
        with self.g.as_default():
            saver = tf.train.Saver(tf.global_variables())
            saver.restore(sess, os.path.join(model_save_path, 'model.ckpt'))
        
    def load_checkpoint(self, checkpoint_path):
        sess = self.sess
        with self.g.as_default():
            saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(checkpoint_path)
        print('loading model', ckpt.model_checkpoint_path)
        tf.logging.info('Loading model %s.', ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)


        '''x = tf.layers.conv2D(x, filters1, (1, 1), use_bias=False,
                             kernel_initializer='he_normal',
                             kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                             name=bn_name_base + '2a')'''
    
    def vpp(self, img_input, drop_rate, is_training, batch_size=None):
        ####################### IMG CONV #######################
        x = img_input
        conv_name_base = 'img_conv1_'
        x = layers.Conv2D(16, (9, 9), padding='valid', use_bias=False,
                          kernel_initializer='he_normal',
                          kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                          name=conv_name_base + '1')(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(16, (5, 5), padding='valid', use_bias=False,
                          kernel_initializer='he_normal',
                          kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                          name=conv_name_base + '2')(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
                
        conv_name_base = 'img_conv2_'
        x = layers.Conv2D(16, (5, 5), padding='valid', use_bias=False,
                          kernel_initializer='he_normal',
                          kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                          name=conv_name_base + '1')(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(16, (5, 5), padding='valid', use_bias=False,
                          kernel_initializer='he_normal',
                          kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                          name=conv_name_base + '2')(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        
        conv_name_base = 'img_conv3_'
        x = layers.Conv2D(128, (17, 17), padding='valid', use_bias=False,
                          kernel_initializer='he_normal',
                          kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                          name=conv_name_base + '1')(x)
        x = layers.Activation('relu')(x)
        
        ####################### TO NET OUTPUT ########################
        mlp_name_base = 'mlp_'
        x = layers.Flatten()(x)
        x = layers.Dense(128, use_bias=False, kernel_initializer='he_normal', 
                         kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                         name=mlp_name_base + '1')(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(drop_rate)(x)
        x = layers.Dense(self._pose_dim, use_bias=False, kernel_initializer='he_normal', 
                         kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                         name=mlp_name_base + '2')(x)
        X = layers.Activation('tanh')(x)
        return x
    
    def vpp_mixture(self, img_input, drop_rate, is_training, batch_size=None):
        ####################### IMG CONV #######################
        x = img_input
        conv_name_base = 'img_conv1_'
        x = layers.Conv2D(16, (3, 3), padding='valid', use_bias=False,
                          kernel_initializer='he_normal',
                          kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                          name=conv_name_base + '1')(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(16, (3, 3), padding='valid', use_bias=False,
                          kernel_initializer='he_normal',
                          kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                          name=conv_name_base + '2')(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
                
        conv_name_base = 'img_conv2_'
        x = layers.Conv2D(16, (3, 3), padding='valid', use_bias=False,
                          kernel_initializer='he_normal',
                          kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                          name=conv_name_base + '1')(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(16, (3, 3), padding='valid', use_bias=False,
                          kernel_initializer='he_normal',
                          kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                          name=conv_name_base + '2')(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        
        conv_name_base = 'img_conv3_'
        x = layers.Conv2D(128, (5, 5), padding='valid', use_bias=False,
                          kernel_initializer='he_normal',
                          kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                          name=conv_name_base + '1')(x)
        x = layers.Activation('relu')(x)
        
        ####################### TO NET OUTPUT ########################
        mlp_name_base = 'mlp_'
        x = layers.Flatten()(x)
        x = layers.Dense(256, use_bias=False, kernel_initializer='he_normal', 
                         kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                         name=mlp_name_base + '1')(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(drop_rate)(x)
        x = layers.Dense(128, use_bias=False, kernel_initializer='he_normal', 
                         kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                         name=mlp_name_base + '1')(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(drop_rate)(x)
        w_and_mu_and_logsig = layers.Dense(self._num_mixtures * (1 + 2 * self._pose_dim), use_bias=False, kernel_initializer='he_normal',
                                           kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                                           name='mlp_w_and_mu_and_logvar')(x)
        w_and_mu_and_logsig = tf.reshape(w_and_mu_and_logsig, (-1, self._num_mixtures, 1 + 2 * self._pose_dim))
        log_ws = w_and_mu_and_logsig[..., 0]
        mus = w_and_mu_and_logsig[..., 1:1 + self._pose_dim]
        log_sigs = w_and_mu_and_logsig[..., 1 + self._pose_dim:1 + 2 * self._pose_dim]
        log_sigs = tf.clip_by_value(log_sigs, LOG_SIG_CAP_MIN, LOG_SIG_CAP_MAX)
        
        return log_ws, mus, log_sigs
    
    def create_log_gaussian(self, mu, log_sig, t, batch_size=None):
        normalized_dist_t = (t - mu) * tf.exp(-log_sig)  # ... x D
        quadratic = - 0.5 * tf.reduce_sum(normalized_dist_t ** 2, axis=-1) # ... x (None)

        log_z = tf.reduce_sum(log_sig, axis=-1)  # ... x (None)
        D_t = tf.cast(tf.shape(mu)[-1], tf.float32)
        log_z += 0.5 * D_t * np.log(2 * np.pi)

        log_p = quadratic - log_z
        return log_p