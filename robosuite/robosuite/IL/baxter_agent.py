import os
from behavior_cloning import * #SimpleCNN

import pickle
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer('render', 0, 'render the screens')
flags.DEFINE_integer('num_episodes', 10000, 'number of episodes')
flags.DEFINE_integer('use_feature', 0, 'using feature-base states or image-base states.')
flags.DEFINE_string('task', 'reach', 'name of task: [ reach / push / pick ]')
flags.DEFINE_string('action_type', '2D', '[ 2D / 3D ]')

flags.DEFINE_integer('save_data', 1, 'save data or not')
flags.DEFINE_integer('max_buff', 2000, 'number of steps saved in one data file.')

flags.DEFINE_string('model_type', 'greedy', 'greedy / bc')
flags.DEFINE_string('model_name', 'reach_0811_191540', 'name of the trained BC model')

FLAGS = flags.FLAGS
using_feature = (FLAGS.use_feature==1)
if using_feature:
    print('This agent will use feature-based states..!!')
else:
    print('This agent will use image-based states..!!')

render = bool(FLAGS.render)
save_data = bool(FLAGS.save_data)
print_on = False
task = FLAGS.task
action_type = FLAGS.action_type

if FLAGS.model_type=='bc':
    render = True
    save_data = False
    print_on = True
    assert FLAGS.model_name is not None


# camera resolution
screen_width = 192 #64
screen_height = 192 #64
crop = 128

# Path which data will be saved in.
save_name = os.path.join(FILE_PATH, 'data')

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
    env = BaxterEnv(env, task=task, render=render, using_feature=using_feature, rgbd=True, action_type=action_type)

    if FLAGS.model_type=='greedy':
        agent = GreedyAgent(env)
    elif FLAGS.model_type=='bc':
        trained_model = SimpleCNN(task, env.action_size, model_name=FLAGS.model_name)
        agent = BCAgent(trained_model)

    if not os.path.exists(save_name):
        os.makedirs(save_name)

    total_steps = 0
    success_log = []
    buff_states = []
    buff_actions = []
    for n in range(FLAGS.num_episodes):
        if print_on:
            print('[Episode %d]'%n)
        obs = env.reset()
        done = False
        cumulative_reward = 0.0
        step_count = 0
        ep_buff_states = []
        ep_buff_actions = []

        while not done:
            step_count += 1
            action = agent.get_action(obs)
            new_obs, reward, done, _ = env.step(action)
            total_steps += 1
            if print_on:
                print('action: %d / reward: %.2f'%(action, reward))
            # print(step_count, 'steps \t action: ', action, '\t reward: ', reward)
            cumulative_reward += reward

            ep_buff_states.append(obs)
            ep_buff_actions.append(action)
            obs = new_obs

        success = (cumulative_reward >= 90)
        success_log.append(int(success))
        if success:
            buff_states += ep_buff_states
            buff_actions += ep_buff_actions

            # recording the trajectories
            if save_data:
                if not os.path.isdir(save_name):
                    os.makedirs(save_name)
                if len(buff_states) >= FLAGS.max_buff:
                    f_list = os.listdir(save_name)
                    num_pickles = len([f for f in f_list if task in f])
                    save_num = num_pickles // 2
                    with open(os.path.join(save_name, task + '_s_%d.pkl'%save_num), 'wb') as f:
                        pickle.dump(np.array(buff_states)[:FLAGS.max_buff], f)
                    with open(os.path.join(save_name, task + '_a_%d.pkl'%save_num), 'wb') as f:
                        pickle.dump(np.array(buff_actions)[:FLAGS.max_buff], f)
                    print(save_num, '-th file saved.')
                    buff_states = buff_states[FLAGS.max_buff:]
                    buff_actions = buff_actions[FLAGS.max_buff:]
                    # buff_states, buff_actions = [], []

        if print_on:
            print('success rate?:', np.mean(success_log), success_log)
        print('Episode %d ends.'%(n+1), '( Total steps:', total_steps, ')')
        print('Ep len:', step_count, 'steps.   Ep reward:', cumulative_reward)
        print()


class GreedyAgent():
    def __init__(self, env, action_type='2D'):
        self.env = env
        self.task = self.env.task
        # self.using_feature = self.env.using_feature
        self.mov_dist = self.env.mov_dist
        self.action_size = self.env.action_size
        self.action_type = action_type

    def get_action(self, obs):
        mov_dist = self.mov_dist
        if self.task == 'reach':
            predicted_distance_list = []
            for action in range(self.action_size):
                if action < 8:
                    mov_degree = action * np.pi / 4.0
                    arm_pos = self.env.arm_pos + np.array([mov_dist * np.cos(mov_degree), mov_dist * np.sin(mov_degree), 0.0])
                elif action == 8:
                    arm_pos = self.env.arm_pos + np.array([0.0, 0.0, mov_dist])
                elif action == 9:
                    arm_pos = self.env.arm_pos + np.array([0.0, 0.0, -mov_dist])

                if arm_pos[2] < 0.57:
                    predicted_distance_list.append(np.inf)
                else:
                    dist = np.linalg.norm(arm_pos - self.env.obj_pos)
                    predicted_distance_list.append(dist)

            action = np.argmin(predicted_distance_list)

        elif self.task == 'push':
            vec_target_obj = self.env.target_pos - self.env.obj_pos
            vec_obj_arm = self.env.obj_pos - self.env.arm_pos
            mov_vec_list = []
            for a in range(8):
                mov_degree = a * np.pi / 4.0
                mov_vec_list.append(np.array([mov_dist * np.cos(mov_degree), mov_dist * np.sin(mov_degree)]))
                # elif a == 8:
                #     mov_vec_list.append(np.array([0.0, 0.0, mov_dist]))
                # elif a == 9:
                #     mov_vec_list.append(np.array([0.0, 0.0, -mov_dist]))
            mov_cos_list = [self.get_cos(v, vec_obj_arm[:2]) for v in mov_vec_list]

            if self.get_cos(vec_target_obj[:2], vec_obj_arm[:2]) > 0:
                if self.env.arm_pos[2] < 0.65:
                    action = np.argmax(mov_cos_list)
                else:
                    if np.linalg.norm(vec_obj_arm) > 2.0 * mov_dist:
                        action = 9
                    else:
                        next_obj_arm = [vec_obj_arm[:2] - v for v in mov_vec_list]
                        next_cos_list = [self.get_cos(vec_target_obj[:2], w) for w in next_obj_arm]
                        action = np.argmax(next_cos_list)
                        '''
                        best_a = np.argmax(mov_cos_list)
                        mov_cos_list[best_a] = np.min(mov_cos_list)
                        next_best_a = np.argmax(mov_cos_list)
                        if self.get_cos(mov_vec_list[best_a][:2], vec_target_obj[:2]) > 0:
                            action = best_a
                        else:
                            action = next_best_a
                        action = (action + 4) % 8
                        '''
            else:
                if self.env.arm_pos[2] < 0.65:
                    action = 8
                else:
                    next_obj_arm = [vec_obj_arm[:2] - v for v in mov_vec_list]
                    next_cos_list = [self.get_cos(vec_target_obj[:2], w) for w in next_obj_arm]
                    action = np.argmax(next_cos_list)

        return action

    def get_cos(self, vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


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


