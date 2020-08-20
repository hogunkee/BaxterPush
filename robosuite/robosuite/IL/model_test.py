import os
from behavior_cloning import * #SimpleCNN

import pickle
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer('render', 1, 'render the screens')
flags.DEFINE_integer('num_episodes', 100, 'number of episodes')
flags.DEFINE_integer('use_feature', 0, 'using feature-base states or image-base states.')
flags.DEFINE_string('task', 'reach', 'name of task: [ reach / push / pick ]')
flags.DEFINE_string('action_type', '2D', '[ 2D / 3D ]')
flags.DEFINE_integer('random_spawn', 0, 'robot arm random pos or not (only for REACH task)')

flags.DEFINE_string('model_name', 'reach_0811_191540', 'name of the trained BC model')

FLAGS = flags.FLAGS
using_feature = bool(FLAGS.use_feature)
if using_feature:
    print('This agent will use feature-based states..!!')
else:
    print('This agent will use image-based states..!!')

render = bool(FLAGS.render)
print_on = True #False
task = FLAGS.task
action_type = FLAGS.action_type
random_spawn = bool(FLAGS.random_spawn)

assert FLAGS.model_name is not None


# camera resolution
screen_width = 192 #64
screen_height = 192 #64
crop = 128

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU is not efficient here

def main():
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
    env = BaxterEnv(env, task=task, render=render, using_feature=using_feature, random_spawn=random_spawn, rgbd=True, action_type=action_type)

    policynet = SimpleCNN(task, env.action_size, model_name=FLAGS.model_name)
    agent = BCAgent(policynet)

    # policynet.set_env(env)
    # policynet.test_agent(agent.sess)
    # print('-------------------stop point------------------------')

    total_steps = 0
    success_log = []
    for n in range(FLAGS.num_episodes):
        if print_on:
            print('[Episode %d]'%n)
        obs = env.reset()
        done = False
        cumulative_reward = 0.0
        ep_len = 0
        ep_buff_states = []
        ep_buff_actions = []

        while not done:
            ep_len += 1
            action = agent.get_action(obs)
            obs, reward, done, _ = env.step(action)
            if print_on:
                print('action: %d / reward: %.2f'%(action, reward))
            # print(step_count, 'steps \t action: ', action, '\t reward: ', reward)
            cumulative_reward += reward

            ep_buff_states.append(obs)
            ep_buff_actions.append(action)

        success = bool(cumulative_reward >= 90)
        success_log.append(int(success))

        # check action distribution #
        if True:
            print('---' * 10)
            print('action distribution:')
            for a in range(max(env.action_size, max(ep_buff_actions)+1)):
                print('%d: %.2f'%(a, list(ep_buff_actions).count(a)/len(ep_buff_actions)))
            print('---' * 10)

        print('success rate?:', np.mean(success_log), success_log[-10:])
        print('Episode %d ends.'%(n+1))
        print('Ep len:', ep_len, 'steps.   Ep reward:', cumulative_reward)
        print()


class BCAgent():
    def __init__(self, BC_model):
        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=gpu_config)
        self.model = BC_model
        self.model.load_model(self.sess)

    def get_action(self, obs):
        clipped_obs = np.clip(obs, 0.0, 5.0)
        action = self.sess.run(self.model.q_action, feed_dict={self.model.s_t: [clipped_obs]})
        return action


if __name__=='__main__':
    main()


