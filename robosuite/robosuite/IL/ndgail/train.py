import sys
import os
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '..', 'client'))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dobroEnv
import gym

from nets.ppo2 import Agent
from nets.discriminator import Discriminator
#from graph_drawer import Graph
from dobroEnv.util import run_episodes2, random_seed
from utils import get_total_parameter_num
from utils import ArgParser
from utils import ROOT_RANK
from utils import Logger

from tensorboardX import SummaryWriter
from pybullet_utils.arg_parser import ArgParser

from collections import deque
from copy import deepcopy
import tensorflow as tf
from mpi4py import MPI
import numpy as np
import pickle
import random
import math
import time
import datetime

from client.env import CarlaEnv
from client.observation_utils import CarlaObservationConverter
from client.action_utils import CarlaActionsConverter
def constraint_reward(states, actions, e_states, e_actions, f_states, f_actions, alpha=3.0):
    rewards = []
    for state, action in zip(states, actions):
        s = np.mean(np.square(np.tile([state],(len(e_states),1,1,1)) - e_states))
        #a = np.mean(np.square(np.tile(action,(len(e_actions),1)) - e_actions))
        #dist_e = np.sqrt(s+a)
        dist_e = np.sqrt(s)
        s = np.mean(np.square(np.tile([state],(len(f_states),1,1,1)) - f_states))
        #a = np.mean(np.square(np.tile(action,(len(f_actions),1)) - f_actions))
        #dist_f = np.sqrt(s+a)
        dist_f = np.sqrt(s)
        d_e = (1 + dist_e/alpha)**((1+alpha)/2)
        d_f = (1 + dist_f/alpha)**((1+alpha)/2)
        rewards.append(d_f/(d_e+d_f))
    return rewards


def main(is_train=True, is_record=True):
    RANK = MPI.COMM_WORLD.Get_rank()
    NUM_PROC = MPI.COMM_WORLD.Get_size()
    is_root = RANK == ROOT_RANK

    if not is_root:
        is_record = False

    if len(sys.argv) == 2:
        seed = int(sys.argv[1])
    else:
        seed = 1337
    print('seed :', seed)
    random_seed(seed=seed + RANK) #for random seed
    now = datetime.datetime.now()
    print(now)

    #with open('../trajectory/expert_s.pkl', 'rb') as f:
    with open(os.path.join(FILE_PATH, '..', 'data/expert_s.pkl'), 'rb') as f:
        expert_states = pickle.load(f)
        expert_states = np.array(expert_states)
    #with open('../trajectory/expert_a.pkl', 'rb') as f:
    with open(os.path.join(FILE_PATH, '..', 'data/expert_a.pkl'), 'rb') as f:
        expert_actions = pickle.load(f)
        expert_actions = np.array(expert_actions)
    #with open('../trajectory/fail_s.pkl', 'rb') as f:
    with open(os.path.join(FILE_PATH, '..', 'data/fail_s.pkl'), 'rb') as f:
        fail_states = pickle.load(f)
        fail_states = np.array(fail_states)
    #with open('../trajectory/fail_a.pkl', 'rb') as f:
    with open(os.path.join(FILE_PATH, '..', 'data/fail_a.pkl'), 'rb') as f:
        fail_actions = pickle.load(f)
        fail_actions = np.array(fail_actions)

    # demo's num
    data_num = len(expert_states)
    e_data_num = data_num//2
    f_data_num = data_num - e_data_num
    expert_states, expert_actions = expert_states[:e_data_num], expert_actions[:e_data_num]
    fail_states, fail_actions = fail_states[:f_data_num], fail_actions[:f_data_num]
    
    ################################ configuration #############################################
    arg_parser = ArgParser()
    assert arg_parser.load_file('args.txt')
    # env_name = arg_parser.parse_string('env_name')
    # env = gym.make(env_name)

    obs_converter = CarlaObservationConverter(h=84, w=84, rel_coord_system=False)
    action_converter = CarlaActionsConverter('continuous')
    env = CarlaEnv(obs_converter, action_converter, 0, reward_class_name='CarlaReward', port=2000, subset='gail', #'keep_lane',\
                   video_every=10, video_dir=os.path.join(FILE_PATH, 'video', now.strftime("%m%d_%H%M%S"))) # 'CIRLReward'
    agent = Agent(env, arg_parser, RANK)
    disc = Discriminator(env, arg_parser, expert_states, expert_actions, RANK)
    if is_record:
        logger = Logger(arg_parser)

    writer = SummaryWriter(os.path.join(FILE_PATH, 'log/', 'tensorboard', now.strftime("%m%d_%H%M%S")))

    maxstep = arg_parser.parse_int('max_step') // NUM_PROC #parallel computing
    iters = arg_parser.parse_int('total_iters')
    print_period = arg_parser.parse_int('print_period')
    game_title = arg_parser.parse_string('game_title')
    start_train = arg_parser.parse_int('start_train')
    BATCH_SIZE = arg_parser.parse_int('batch_size')
    save_name = arg_parser.parse_string('save_name')
    gamma = arg_parser.parse_float('gamma')
    ETA = arg_parser.parse_float('eta')
    ETA_DECAY =  arg_parser.parse_float('eta_decay')
    max_reward = -10000000

    if is_train and is_root: 
        print('total parameter\'s num : {}'.format(get_total_parameter_num()))
        # graph = Graph(freq=1000, title=game_title.upper(), label=save_name)
    ################################ configuration #############################################


    avg_true_return_list = deque(maxlen=print_period)
    avg_return_list = deque(maxlen=print_period)
    avg_pol_loss_list = deque(maxlen=print_period)
    avg_disc_val_loss_list = deque(maxlen=print_period)
    avg_const_val_loss_list = deque(maxlen=print_period)
    avg_agent_acc_list = deque(maxlen=print_period)
    avg_expert_acc_list = deque(maxlen=print_period)

    if is_root:
        print("\n###########################")
        print("{} Experiment Start. # of processors : {}".format(game_title, NUM_PROC))

    episode = 0
    for iter in range(iters):
        start_t = time.time()
        episodes, trajectories = run_episodes2(env, agent, episode, is_train, is_root, maxstep, ETA, logger, is_record, pos=19, weather=0)
        states = np.concatenate([t[0] for t in trajectories])
        actions = np.concatenate([t[1] for t in trajectories])
        true_rewards = np.concatenate([t[6] for t in trajectories])
        episode += episodes

        if is_train:
            if np.sum(true_rewards)/episodes > max_reward:
                max_reward = np.sum(true_rewards)/episodes
                if  iter >= start_train + 1 and is_root:
                    print("#"*50)
                    print("Winner!! Now we save model")
                    agent.save()
                    disc.save()
            ## check! ##
            print('root?:', is_root)
            if is_root:
                #disc_replay_buffer_s += deque(states)
                #disc_replay_buffer_a += deque(actions)
                #a_acc, e_acc, disc_entropy, a_loss, e_loss, gp_loss = disc.train(states, actions, disc_replay_buffer_s, disc_replay_buffer_a, 64)
                a_acc, e_acc, disc_entropy, a_loss, e_loss, gp_loss = disc.train(states, actions, BATCH_SIZE)
                dist = disc.get_measure_distance(states, actions)
                print("[DISC] a_loss : {}, e_loss : {}, gp_loss : {}".format(a_loss, e_loss, gp_loss))

                #agent train data preprocess
                disc_targets = []
                const_targets = []
                gaes = []
                score = 0
                for t in trajectories:
                    disc_rewards = disc.get_reward(t[0], t[1])
                    const_rewards = constraint_reward(t[0],t[1], expert_states, expert_actions, fail_states, fail_actions)
                    rewards = np.array(disc_rewards) + ETA*np.array(const_rewards)
                    next_values = t[3] + ETA*t[5]
                    values = t[2] + ETA*t[4]
                    #rewards = (1-ETA)*np.array(disc_rewards) + ETA*np.array(const_rewards)
                    #next_values = (1-ETA)*t[3] + ETA*t[5]
                    #values = (1-ETA)*t[2] + ETA*t[4]
                    gae = agent.get_gaes(rewards, next_values, values)
                    disc_target = np.zeros_like(disc_rewards)
                    const_target = np.zeros_like(const_rewards)
                    disc_ret = 0
                    const_ret = 0
                    for t in reversed(range(len(disc_rewards))):
                        disc_ret = disc_rewards[t] + agent.discount_factor*disc_ret
                        const_ret = const_rewards[t] + agent.discount_factor*const_ret
                        disc_target[t] = disc_ret
                        const_target[t] = const_ret
                    gae = np.array(gae).astype(dtype=np.float32)
                    disc_target = np.array(disc_target).astype(dtype=np.float32)
                    const_target = np.array(const_target).astype(dtype=np.float32)
                    gaes.append(gae)
                    disc_targets.append(disc_target)
                    const_targets.append(const_target)
                    score += np.sum(rewards)

                # convert list to numpy array for feeding tf.placeholder
                gaes = np.concatenate(gaes)
                disc_targets = np.concatenate(disc_targets)
                const_targets = np.concatenate(const_targets)
                #마운틴카에서 gaes를 노멀라이즈했을때 좀더 안정적으로 수렴하는 느낌이 들었음
                gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-6)

                #for eta decay
                first_state = states[0]
                _, first_disc_value, first_const_value = agent.get_action(first_state, True)

                inp = [states, actions, gaes, disc_targets, const_targets]
                if iter >= start_train:
                    loss_p, loss_disc_v, loss_const_v, kl, entropy = agent.train(inp, batch_size=BATCH_SIZE)
                else:
                    loss_p, loss_disc_v, loss_const_v, kl, entropy = agent.train(inp, batch_size=BATCH_SIZE, only_value=True)

                avg_pol_loss_list.append(loss_p)
                avg_disc_val_loss_list.append(loss_disc_v)
                avg_const_val_loss_list.append(loss_const_v)
                avg_true_return_list.append(np.sum(true_rewards)/episodes)
                avg_return_list.append(score/episodes)
                avg_agent_acc_list.append(a_acc)
                avg_expert_acc_list.append(e_acc)

                writer.add_scalar('train/mean_true_return', np.mean(avg_true_return_list), iter)
                writer.add_scalar('train/mean_agent_accuracy', np.mean(avg_agent_acc_list), iter)
                writer.add_scalar('train/mean_expert_accuracy', np.mean(avg_expert_acc_list), iter)
                writer.add_scalar('train/dist', dist, iter)
                writer.add_scalar('train/eta', ETA, iter)
                #for eta decay
                _, post_first_disc_value, post_first_const_value = agent.get_action(first_state, True)
                disc_value_change = post_first_disc_value - first_disc_value
                const_value_change = post_first_const_value - first_const_value
                if const_value_change < disc_value_change:
                    ETA *= ETA_DECAY

                # graph.update(np.mean(avg_true_return_list), kl, np.mean(avg_agent_acc_list), np.mean(avg_expert_acc_list), dist)

                if (iter+1)%print_period==0:
                    print('[{}/{}]true return: {:.3f}, return : {:.3f}, disc_value loss : {:.3f}, const_value loss : {:.3f}, policy loss : {:.3f}, policy kl : {:.5f}, policy entropy : {:.3f}, a_acc : {:.3f}, e_acc : {:.3f}'.\
                        format(iter, iters, np.mean(avg_true_return_list), np.mean(avg_return_list), np.mean(avg_disc_val_loss_list), np.mean(avg_const_val_loss_list), np.mean(avg_pol_loss_list), kl, entropy, np.mean(avg_agent_acc_list), np.mean(avg_expert_acc_list)))
                    print("training 하기 :",time.time() - start_t)

                if is_record and iter >= start_train:
                    logger.write(1, [np.mean(avg_true_return_list), np.mean(avg_agent_acc_list), np.mean(avg_expert_acc_list), kl, dist, a_loss, e_loss, gp_loss, np.mean(avg_disc_val_loss_list), np.mean(avg_const_val_loss_list), np.mean(avg_pol_loss_list), disc_value_change, const_value_change])
                    logger.save()

    if is_root: 
        # graph.update(0, 0, 0, 0, 0, finished=True)
        print("Finish.")
    

if __name__ == "__main__":
    main(is_train=True, is_record=True)

