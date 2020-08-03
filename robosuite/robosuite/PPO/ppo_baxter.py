import os
import sys
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '../..'))
sys.path.append(os.path.join(FILE_PATH, '..', 'scripts'))
from demo_baxter_rl_pushing import *

import shutil
import time
from time import sleep

from ppo.renderthread import RenderThread
from ppo.models import *
from ppo.trainer import Trainer

import datetime
import tensorflow as tf
flags = tf.app.flags

# ## Proximal Policy Optimization (PPO)
# Contains an implementation of PPO as described [here](https://arxiv.org/abs/1707.06347).

# Algorithm parameters
# batch-size=<n>           How many experiences per gradient descent update step [default: 64].
batch_size = 512
# beta=<n>                 Strength of entropy regularization [default: 2.5e-3].
beta = 2.5e-3
# buffer-size=<n>          How large the experience buffer should be before gradient descent [default: 2048].
buffer_size = batch_size * 4
# epsilon=<n>              Acceptable threshold around ratio of old and new policy probabilities [default: 0.2].
epsilon = 0.2
# gamma=<n>                Reward discount rate [default: 0.99].
gamma = 0.99
# hidden-units=<n>         Number of units in hidden layer [default: 64].
hidden_units = 128
# lambd=<n>                Lambda parameter for GAE [default: 0.95].
lambd = 0.95
# learning-rate=<rate>     Model learning rate [default: 3e-4].
learning_rate = 1e-3 #3e-4 #4e-5
# max-steps=<n>            Maximum number of steps to run environment [default: 1e6].
max_steps = 3e5 #15e6
# normalize                Activate state normalization for this many steps and freeze statistics afterwards.
normalize_steps = 0
# num-epoch=<n>            Number of gradient descent steps per batch of experiences [default: 5].
num_epoch = 10
# num-layers=<n>           Number of hidden layers between state/observation and outputs [default: 2].
num_layers = 1
# time-horizon=<n>         How many steps to collect per agent before adding to buffer [default: 2048].
time_horizon = 128 #512 #2048

# General parameters
# keep-checkpoints=<n>     How many model checkpoints to keep [default: 5].
keep_checkpoints = 5

# run-path=<path>          The sub-directory name for model and summary statistics.
file_path = os.path.dirname(os.path.abspath(__file__))
summary_path = os.path.join(file_path, 'PPO_summary')
model_path = os.path.join(file_path, 'models')
# summary-freq=<n>         Frequency at which to save training statistics [default: 10000].
summary_freq = 500 #100 #buffer_size * 5
# save-freq=<n>            Frequency at which to save model [default: 50000].
save_freq = 2000 #500 #summary_freq

flags.DEFINE_integer('use_feature', 0, 'using feature-base states or image-base states.')
flags.DEFINE_integer('train', 1, 'Train a new model or test the trained model.')
flags.DEFINE_integer('cutout', 1, 'random cutout - prob 0.7 (default)')
flags.DEFINE_integer('continuous', 1, 'continuous / discrete action space')
flags.DEFINE_string('model_name', None, 'name of trained model')
flags.DEFINE_string('task', 'reach', 'name of task: reach / push / pick')

FLAGS = flags.FLAGS
using_feature = bool(FLAGS.use_feature)
if using_feature:
    print('This model will use feature-based states..!!')
else:
    print('This model will use image-based states..!!')

if FLAGS.train==1:
    load_model = False
    render = False
    train_model = True
else:
    load_model = True
    render = True
    train_model = False

task = FLAGS.task
cutout = bool(FLAGS.cutout)
continuous = bool(FLAGS.continuous)

if FLAGS.model_name:
    model_path = os.path.join(model_path, FLAGS.model_name)
    summary_path = os.path.join(summary_path, FLAGS.model_name)
    assert task in FLAGS.model_name
else:
    now = datetime.datetime.now()
    # base = 'fb' if using_feature else 'ib'
    base = 'con' if continuous else 'dis'
    model_path = os.path.join(model_path, task + '_' + base + '_' + now.strftime("%m%d_%H%M%S"))
    summary_path = os.path.join(summary_path, task + '_' + base + '_' + now.strftime("%m%d_%H%M%S"))

# load                     Whether to load the model or randomly initialize [default: False].
# load_model = True #False #True
# train                    Whether to train model, or only run inference [default: False].
# train_model = False #True
# render environment to display progress
# render = True #False #True
# save recordings of episodes
record = True

# Baxter parameters
# camera resolution
screen_width = 64 #96
screen_height = 64 #96
crop = None #64 #None

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU is not efficient here

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
    crop=crop
)
env = IKWrapper(env)
env = BaxterEnv(env, task=task, continuous=continuous, render=render, using_feature=using_feature)

tf.reset_default_graph()

ppo_model = create_agent_model(env, is_continuous=continuous, lr=learning_rate,
                               h_size=hidden_units, epsilon=epsilon,
                               beta=beta, max_step=max_steps,
                               normalize=normalize_steps, num_layers=num_layers,
                               use_states=using_feature)

# use_observations = True
# use_states = False

if not load_model:


def random_quat():
    rand = np.random.rand(3)
    r1 = np.sqrt(1.0 - rand[0])
    r2 = np.sqrt(rand[0])
    pi2 = np.pi * 2.0
    t1 = pi2 * rand[1]
    t2 = pi2 * rand[2]
    return np.array((np.sin(t1) * r1, np.cos(t1) * r1, np.sin(t2) * r2, np.cos(t2) * r2), dtype=np.float32)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--seed', type=int, default=0)
    parser.add_argument(
        '--num-objects', type=int, default=2)
    parser.add_argument(
        '--num-episodes', type=int, default=10000)
    parser.add_argument(
        '--num-steps', type=int, default=1)
    parser.add_argument(
        '--render', type=bool, default=True)
    parser.add_argument(
        '--bin-type', type=str, default="table") # table, bin, two
    parser.add_argument(
        '--object-type', type=str, default="cube") # T, Tlarge, L, 3DNet, stick, round_T_large
    parser.add_argument(
        '--test', type=bool, default=False)
    parser.add_argument(
        '--config-file', type=str, default="config_example.yaml")
    args = parser.parse_args()

    np.random.seed(args.seed)

    env = robosuite.make(
        "BaxterPush",
        bin_type=args.bin_type,
        object_type=args.object_type,
        ignore_done=True,
        has_renderer=True,
        camera_name="eye_on_right_wrist",
        gripper_visualization=False,
        use_camera_obs=False,
        use_object_obs=False,
        camera_depth=True,
        num_objects=args.num_objects,
        control_freq=100
    )
    env = IKWrapper(env)

    render = args.render

    cam_offset = np.array([0.05, 0, 0.15855])
    #cam_offset = np.array([0.05755483, 0.0, 0.16810357])
    right_arm_camera_id = env.sim.model.camera_name2id("eye_on_right_wrist")
    left_arm_camera_id = env.sim.model.camera_name2id("eye_on_left_wrist")

    arena_pos = env.env.mujoco_arena.bin_abs
    init_pos = arena_pos + np.array([0.0, 0.0, 0.3])
    init_obj_pos = arena_pos + np.array([0.0, 0.0, 0.0])
    float_pos = arena_pos + np.array([0.0, 0.0, 0.3])

    num_episodes = args.num_episodes
    num_steps = args.num_steps
    test = args.test
    save_num = args.seed

    rl_env = BaxterEnv(env, task='pick', render=render)

    success_count, failure_count, controller_failure = 0, 0, 0
    for i in  range(num_episodes):
        state = rl_env.reset()
        done = False
        while not done:
            state, reward, done, _ = rl_env.step(action)
    '''
    for i in range(0, num_episodes):

        rl_env.reset()
        for j in range(12):
            state, reward, done, _ = rl_env.step(j)
    '''

    shutil.rmtree(summary_path, ignore_errors=True)

if not os.path.exists(model_path):
    os.makedirs(model_path)

if not os.path.exists(summary_path):
    os.makedirs(summary_path)

tf.set_random_seed(0) #np.random.randint(1024))
init = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=keep_checkpoints)

gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True

with tf.Session(config=gpu_config) as sess:
    # Instantiate model parameters
    if load_model:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt is None:
            print('The model {0} could not be found. Make sure you specified the right --run-path'.format(model_path))
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(init)

    steps, last_reward = sess.run([ppo_model.global_step, ppo_model.last_reward])
    summary_writer = tf.summary.FileWriter(summary_path)
    obs = env.reset() #[brain_name]
    trainer = Trainer(ppo_model, sess, continuous, using_feature, train_model, cutout=cutout)

    while steps <= max_steps or not train_model:
        if env.global_done:
            obs = env.reset() #[brain_name]
            trainer.reset_buffers(total=True) #({'ppo': None}, total=True)
        # Decide and take an action
        obs = trainer.take_action(obs, env, steps, normalize_steps, stochastic=True)
        trainer.process_experiences(obs, env.global_done, time_horizon, gamma, lambd)
        if len(trainer.training_buffer['actions']) > buffer_size and train_model:
            print("Optimizing...")
            t = time.time()
            # Perform gradient descent with experience buffer
            trainer.update_model(batch_size, num_epoch)
            print("Optimization finished in {:.1f} seconds.".format(float(time.time() - t)))

        if steps % summary_freq == 0 and steps != 0 and train_model:
            # Write training statistics to tensorboard.
            trainer.write_summary(summary_writer, steps)
        if steps % save_freq == 0 and steps != 0 and train_model:
            # Save Tensorflow model
            save_model(sess=sess, model_path=model_path, steps=steps, saver=saver)
        if train_model:
            steps += 1
            sess.run(ppo_model.increment_step)
            if len(trainer.stats['cumulative_reward']) > 0:
                mean_reward = np.mean(trainer.stats['cumulative_reward'])
                sess.run(ppo_model.update_reward, feed_dict={ppo_model.new_reward: mean_reward})
                last_reward = sess.run(ppo_model.last_reward)

    # Final save Tensorflow model
    if steps != 0 and train_model:
        save_model(sess=sess, model_path=model_path, steps=steps, saver=saver)
#env.close()
export_graph(model_path, env_name="BaxterEnv")
os.system("shutdown")

