import numpy as np
import math
import json
import tensorflow as tf
import tensorflow.contrib.framework as tcf
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
EPS = 1e-8

def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()
    
class FCGQCNN(object):
    def __init__(self, dataset_dir, angular_bins=[8, 4, 8], is_training=False):
        self._is_training = is_training
        self._momentum_rate = 0.9
        self._max_global_grad_norm = 100000000000
        self._train_l2_regularizer = 0.0005
        
        _im_depth_sub_mean_filename = os.path.join(dataset_dir, 'im_depth_sub_mean.npy')
        _im_depth_sub_std_filename = os.path.join(dataset_dir, 'im_depth_sub_std.npy')
        self._im_depth_sub_mean = np.load(_im_depth_sub_mean_filename)
        self._im_depth_sub_std = np.load(_im_depth_sub_std_filename)
        
        self._batch_size = 64
        self._im_height = 100
        self._im_width = 100
        self._im_channels = 1
        self._pose_dim = 1
        self._angular_bins = angular_bins
        self._total_bins = self._angular_bins[0] * self._angular_bins[1] * self._angular_bins[2]
        
        self._weights = {}
        self._optimize_base_layers = True
        
    def load_pretrained_model(self, model_dir=None):
        if model_dir == None:
            self.g = tf.Graph()
            self._build_graph()
            self._init_session()
        else:
            self.g = tf.Graph()
            self._init_weights_file(model_dir)
            self._build_graph()
            self._init_session()
    
    def _build_graph(self):
        with self.g.as_default():
            self.input_im_node = tf.placeholder(tf.float32, shape=[None, None,None,1])#self._im_height, self._im_width, self._im_channels])
            self.input_pred_mask_node = tf.placeholder(tf.int32, shape=[None, 2 * self._total_bins])
            self.input_label_node = tf.placeholder(tf.int32, shape=[None])
            self.learning_rate = tf.placeholder_with_default(tf.constant(0.01), ())
            
            def build_conv_layer(input_node, input_height, input_width, input_channels, filter_h, filter_w, num_filt, name, norm=False, pad='VALID'):
                with tf.name_scope(name):
                    if '{}_weights'.format(name) in self._weights.keys():
                        convW = self._weights['{}_weights'.format(name)]
                        convb = self._weights['{}_bias'.format(name)]
                    else:
                        convW_shape = [filter_h, filter_w, input_channels, num_filt]
                        fan_in = filter_h * filter_w * input_channels
                        std = np.sqrt(2.0 / (fan_in))
                        convW = tf.Variable(tf.truncated_normal(convW_shape, stddev=std), name='{}_weights'.format(name))
                        convb = tf.Variable(tf.truncated_normal([num_filt], stddev=std), name='{}_bias'.format(name))
                    
                    convh = tf.nn.conv2d(input_node, convW, strides=[1, 1, 1, 1], padding=pad) + convb
                    convh = tf.nn.relu(convh)
                    
                    out_height = input_height - filter_h + 1
                    out_width = input_width - filter_w + 1
                    return convh, out_height, out_width, num_filt

            def build_fully_conv_layer(input_node, input_height, input_width, input_channels, num_filt, name, final_fc_layer=False):
                if '{}_weights'.format(name) in self._weights.keys():
                    convW = self._weights['{}_weights'.format(name)]
                    convb = self._weights['{}_bias'.format(name)]
                else:
                    convW_shape = [input_height, input_width, input_channels, num_filt]
                    fan_in = 1 * 1 * input_channels
                    std = np.sqrt(2.0 / (fan_in))
                    convW = tf.Variable(tf.truncated_normal(convW_shape, stddev=std), name='{}_weights'.format(name))
                    convb = tf.Variable(tf.truncated_normal([num_filt], stddev=std), name='{}_bias'.format(name))
                    if final_fc_layer:
                        convb = tf.Variable(tf.constant(0.0, shape=[num_filt]), name='{}_bias'.format(name))
                convh = tf.nn.conv2d(input_node, convW, strides=[1, 1, 1, 1], padding="VALID") + convb
                if not final_fc_layer:
                    convh = tf.nn.relu(convh)
                return convh, 1, 1, num_filt
            
            def build_fc_layer(input_node, fan_in, out_size, name, final_fc_layer=False):
                if '{}_weights'.format(name) in self._weights.keys():
                    fcW = self._weights['{}_weights'.format(name)]
                    fcb = self._weights['{}_bias'.format(name)]
                else:
                    std = np.sqrt(2.0 / (fan_in))
                    fcW = tf.Variable(tf.truncated_normal([fan_in, out_size], stddev=std), name='{}_weights'.format(name))
                    if final_fc_layer:
                        fcb = tf.Variable(tf.constant(0.0, shape=[out_size]), name='{}_bias'.format(name))
                    else:
                        fcb = tf.Variable(tf.truncated_normal([out_size], stddev=std), name='{}_bias'.format(name))
                
                if final_fc_layer:
                    fc = tf.matmul(input_node, fcW) + fcb
                else:
                    fc = tf.nn.relu(tf.matmul(input_node, fcW) + fcb)
                    
                return fc, out_size

            with tf.name_scope("im_stream"):
                input_height = self._im_height
                input_width = self._im_width
                input_channels = self._im_channels
                
                im_stream, input_height, input_width, input_channels = build_conv_layer(self.input_im_node, input_height, input_width, input_channels, 9, 9, 16, "conv1_1", pad="VALID")
                im_stream, input_height, input_width, input_channels = build_conv_layer(im_stream, input_height, input_width, input_channels, 5, 5, 16, "conv1_2", pad="VALID")
                im_stream = tf.nn.max_pool(im_stream, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                input_height = input_height / 2
                input_width = input_width / 2
                
                im_stream, input_height, input_width, input_channels = build_conv_layer(im_stream, input_height, input_width, input_channels, 5, 5, 16, "conv2_1", pad="VALID")
                im_stream, input_height, input_width, input_channels = build_conv_layer(im_stream, input_height, input_width, input_channels, 5, 5, 16, "conv2_2", pad="VALID")
                im_stream = tf.nn.max_pool(im_stream, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                input_height = input_height / 2
                input_width = input_width / 2
                
                im_stream, input_height, input_width, input_channels = build_fully_conv_layer(im_stream, input_height, input_width, input_channels, 128, "fc3")
                im_stream, input_height, input_width, input_channels = build_fully_conv_layer(im_stream, input_height, input_width, input_channels, 128, "fc4")
                im_stream, input_height, input_width, input_channels = build_fully_conv_layer(im_stream, input_height, input_width, input_channels, 2 * self._total_bins, "fc5", final_fc_layer=True)
            
            self.net_output = im_stream
            binwise_split_output = tf.split(self.net_output, self._total_bins, axis=-1)
            binwise_split_output_soft = [tf.nn.softmax(s) for s in binwise_split_output]
            self._output_tensor = tf.concat(binwise_split_output_soft, -1)
            
            if self._is_training:
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                
                net_output_vector = tf.reshape(self.net_output, (-1, 2 * self._total_bins))
                log = tf.reshape(tf.dynamic_partition(net_output_vector, self.input_pred_mask_node, 2)[1], (-1, 2))
                self.unregularized_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(_sentinel=None, labels=self.input_label_node, logits=log))
                self.loss = self.unregularized_loss
                
                t_vars = tf.trainable_variables()
                self.regularizers = tf.nn.l2_loss(t_vars[0])
                for var in t_vars[1:]:
                    self.regularizers += tf.nn.l2_loss(var)
                self.loss += self._train_l2_regularizer * self.regularizers
                
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
                #self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, self._momentum_rate)
                gradients, variables = zip(*self.optimizer.compute_gradients(self.loss, var_list=t_vars))
                gradients, global_grad_norm = tf.clip_by_global_norm(gradients, self._max_global_grad_norm)
                self.train_op = self.optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)

            self.init = tf.global_variables_initializer()

            t_vars = tf.trainable_variables()
            self.assign_ops = {}
            for var in t_vars:
                pshape = var.get_shape()
                pl = tf.placeholder(tf.float32, pshape, var.name[:-2]+'_placeholder')
                assign_op = var.assign(pl)
                self.assign_ops[var] = (assign_op, pl)
                
    def _init_weights_file(self, model_dir):
        
        self._base_layer_names = ["conv1_1_weights", "conv1_1_bias", "conv1_2_weights", "conv1_2_bias",
                                  "conv2_1_weights", "conv2_1_bias", "conv2_2_weights", "conv2_2_bias"]
        
        ckpt_file = os.path.join(model_dir, 'model.ckpt')
        
        with self.g.as_default():
            reader = tf.train.NewCheckpointReader(ckpt_file)
            ckpt_vars = tcf.list_variables(ckpt_file)
            full_var_names = []
            short_names = []
            self._weights = {}
            for variable, shape in ckpt_vars:
                full_var_names.append(variable)
                short_names.append(variable.split('/')[-1])
            
            if self._optimize_base_layers == False: # load only base_layers
                for full_var_name, short_name in zip(full_var_names, short_names):
                    if short_name in self._base_layer_names:
                        self._weights[short_name] = tf.Variable(reader.get_tensor(full_var_name), name=full_var_name, trainable=False)
                return
            
            for full_var_name, short_name in zip(full_var_names, short_names): # load all layers
                self._weights[short_name] = tf.Variable(reader.get_tensor(full_var_name), name=full_var_name, trainable=False)
            
    def _init_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config, graph=self.g)
        self.sess.run(self.init)
    
    def close_session(self):
        self.sess.close()

    def predict(self, image_arr, pose_arr):
        print(image_arr.shape)
        print(pose_arr.shape)
        sub_im_arr = image_arr - np.tile(np.reshape(pose_arr, (-1, 1, 1, 1)), (1, image_arr.shape[1], image_arr.shape[2], 1))
        norm_sub_im_arr = (sub_im_arr - self._im_depth_sub_mean) / self._im_depth_sub_std
        # norm_sub_im_arr
        return self.sess.run(self._output_tensor, feed_dict={self.input_im_node: norm_sub_im_arr})
    
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
