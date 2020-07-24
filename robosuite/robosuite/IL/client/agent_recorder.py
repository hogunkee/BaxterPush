import os
import numpy as np
import pickle
import torch

from pybullet_utils.arg_parser import ArgParser
from collections import namedtuple

import agents
from env import CarlaEnv
from envs_manager import make_vec_envs
from observation_utils import CarlaObservationConverter
from action_utils import CarlaActionsConverter

from utils import load_modules

def run_episodes(env, agent, episode, is_train, is_root, maxstep, logger=None, is_record=None):
    print("##### {} Episode start! #####".format(episode))
    step = 0
    episodes = 0
    trajectories = []
    while step < maxstep:
        #print("##### {} Episode start! #####".format(episode))
        episode += 1
        episodes += 1
        states = []
        actions = []
        values = []
        true_rewards = []

        obs = env.reset()
        s_t = obs['img']
        # s_t = env.reset()
        obs_tensor = {'img': torch.tensor([obs['img']]), 'v': torch.tensor([obs['v']])}
        if True: # args.cuda
            obs_tensor['img'] = obs_tensor['img'].float().cuda()
            obs_tensor['v'] = obs_tensor['v'].float().cuda()

        print('img: ', obs_tensor['img'].shape)
        print('v: ', obs_tensor['v'].shape)

        recurrent_hidden_states = torch.zeros([1, 20], dtype=torch.float32).cuda()
        masks = torch.ones([1, 1], dtype=torch.float32).cuda()
        while True :
            # print('obs_tensor:', obs_tensor)
            # print('r h s:', recurrent_hidden_states)
            # print('masks:', masks)
            value, action, action_log_prob, recurrent_hidden_states = agent.act(
                obs_tensor,
                recurrent_hidden_states,
                masks)
            # a_t, value = agent.get_action(s_t, is_train)

            a_t = action[0].cpu().numpy()
            obs1, r_t, done, info = env.step(a_t)
            s_t1 = obs1['img']
            #r_t = get_reward(r_t)

            if (not is_train) and is_root:
                env.render()
                time.sleep(0.01)
            states.append(s_t)
            actions.append(a_t)
            values.append(value)
            true_rewards.append(r_t)
            step += 1
            if done:
                break
            s_t = s_t1

        if (not is_train) and is_root:
            print(np.sum(true_rewards))

        next_values = values[1:]
        next_values.append(0)
        states = np.array(states).astype(dtype=np.float32)
        actions = np.array(actions).astype(dtype=np.float32)
        values = np.array(values).astype(dtype=np.float32)
        next_values = np.array(next_values).astype(dtype=np.float32)
        trajectories.append([states, actions, values, next_values, true_rewards])

        if is_record and logger != None:
            logger.write(2, [len(states), np.sum(true_rewards)])

    return episodes, trajectories


def main():
    iters = 1
    is_train = False
    max_step = 1
    episode_count = 10

    arg_parser = ArgParser()
    assert arg_parser.load_file('/home/scarab5/hogun_codes/carla-dobro/client/recorder_args.txt')
    #model_path = '/home/scarab5/hogun_codes/carla-dobro/client/outputs/model/debug_2020-05-27_01-16-42/0.pth.tar'
    #save_name = '/home/scarab5/hogun_codes/carla-dobro/client/agent_records/'
    model_path = arg_parser.parse_string('model_path')
    save_name = arg_parser.parse_string('save_name')
    port = arg_parser.parse_int('port')

    print('Model load from: {}'.format(model_path))
    assert os.path.isfile(model_path), 'Checkpoint file does not exist'
    checkpoint = torch.load(model_path)
    config_dict = checkpoint['config']
    config = namedtuple('Config', config_dict.keys())(*config_dict.values())

    norm_reward = not config.no_reward_norm
    norm_obs = not config.no_obs_norm

    obs_converter = CarlaObservationConverter(h=84, w=84, rel_coord_system=config.rel_coord_system)
    action_converter = CarlaActionsConverter(config.action_type)

    device = torch.device("cuda:0") # if args.cuda else "cpu")
    # envs = make_vec_envs(obs_converter, action_converter, port, config.seed, config.num_processes,
    #                                 config.gamma, device, config.reward_class, num_frame_stack=1, subset=config.experiments_subset,
    #                                 norm_reward=norm_reward, norm_obs=norm_obs, apply_her=config.num_virtual_goals > 0)

    env = CarlaEnv(obs_converter, action_converter, 0, config.seed, reward_class_name=config.reward_class, port=port)
    # env = CarlaEnv(obs_converter, action_converter, 0, reward_class_name='CarlaReward')

    if config.agent == 'forward':
        agent = agents.ForwardCarla()

    if config.agent == 'a2c':
        agent = agents.A2CCarla(obs_converter,
                                action_converter,
                                config.value_loss_coef,
                                config.entropy_coef,
                                lr=config.lr,
                                eps=config.eps, alpha=config.alpha,
                                max_grad_norm=config.max_grad_norm)

    elif config.agent == 'acktr':
        agent = agents.A2CCarla(obs_converter,
                                action_converter,
                                config.value_loss_coef,
                                config.entropy_coef,
                                lr=config.lr,
                                eps=config.eps, alpha=config.alpha,
                                max_grad_norm=config.max_grad_norm,
                                acktr=True)

    elif config.agent == 'ppo':
        agent = agents.PPOCarla(obs_converter,
                                action_converter,
                                config.clip_param,
                                config.ppo_epoch,
                                config.num_mini_batch,
                                config.value_loss_coef,
                                config.entropy_coef,
                                lr=config.lr,
                                eps=config.eps,
                                max_grad_norm=config.max_grad_norm)

    assert checkpoint is not None, 'checkpoint Error!!'
    load_modules(agent.optimizer, agent.model, checkpoint)



    episodes = 0
    trajectories = []
    print('Recording starts.')
    for i in range(iters):
        episode, trajectory = run_episodes(env, agent, episodes, is_train, False, max_step)
        print(np.sum(trajectory[0][4]))
        trajectories += trajectory
        episodes += episode

    trajectories.sort(key = lambda x: np.sum(x[4]), reverse = True)
    trajectories = trajectories[:episode_count]
    expert_states = np.concatenate([t[0] for t in trajectories])
    expert_actions = np.concatenate([t[1] for t in trajectories])
    if not os.path.isdir(save_name):
        os.makedirs(save_name)
    with open(save_name+'/expert_s.pkl', 'wb') as f:
        pickle.dump(expert_states, f)
    with open(save_name+'/expert_a.pkl', 'wb') as f:
        pickle.dump(expert_actions, f)
    print("save trajectory!")
    print("Finish.")

if __name__ == "__main__":
    main()