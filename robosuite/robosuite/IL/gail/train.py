import sys
import os

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '..', 'client')) #'/home/gun/Desktop/CARLA/carla_dobro/client')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dobroEnv
import gym

from nets.ppo import Agent
from nets.discriminator import Discriminator
#from graph_drawer import Graph
from dobroEnv.util import run_episodes, random_seed
from utils import get_total_parameter_num
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

    # with open('/home/gun/Desktop/CARLA/data/expert_s.pkl', 'rb') as f:
    with open(os.path.join(FILE_PATH, '..', 'data/expert_s.pkl'), 'rb') as f:
        expert_states = pickle.load(f)
        expert_states = np.array(expert_states)
    with open(os.path.join(FILE_PATH, '..', 'data/expert_a.pkl'), 'rb') as f:
        expert_actions = pickle.load(f)
        expert_actions = np.array(expert_actions)

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
    max_reward = -10000000

    if is_train and is_root: 
        print(get_total_parameter_num())
        # graph = Graph(freq=1000, title=game_title.upper(), label=save_name)
    ################################ configuration #############################################

    avg_true_return_list = deque(maxlen=print_period)
    avg_return_list = deque(maxlen=print_period)
    avg_pol_loss_list = deque(maxlen=print_period)
    avg_val_loss_list = deque(maxlen=print_period)
    avg_agent_acc_list = deque(maxlen=print_period)
    avg_expert_acc_list = deque(maxlen=print_period)

    if is_root:
        print("\n###########################")
        print("{} Experiment Start. # of processors : {}".format(game_title, NUM_PROC))

    episode = 0
    for iter in range(iters):
        start_t = time.time()
        episodes, trajectories = run_episodes(env, agent, episode, is_train, is_root, maxstep, logger, is_record, pos=19, weather=0)
        states = np.concatenate([t[0] for t in trajectories])
        actions = np.concatenate([t[1] for t in trajectories])
        true_rewards = np.concatenate([t[4] for t in trajectories])
        episode += episodes

        if is_train:
            if np.sum(true_rewards)/episodes > max_reward:
                max_reward = np.sum(true_rewards)/episodes
                if  iter >= start_train + 1 and is_root:
                    print("#"*50)
                    print("Winner!! Now we save model")
                    agent.save()
                    disc.save()

            if is_root:
                a_acc, e_acc, disc_entropy, a_loss, e_loss, gp_loss = disc.train(states, actions, BATCH_SIZE)
                dist = disc.get_measure_distance(states, actions)
                rewards = [disc.get_reward(t[0], t[1]) for t in trajectories]
                print("[DISC] a_loss : {}, e_loss : {}, gp_loss : {}".format(a_loss, e_loss, gp_loss))

                gaes, targets = agent.get_gaes_targets(trajectories, rewards)
                inp = [states, actions, gaes, targets]
                if iter >= start_train:
                    loss_p, loss_v, kl, entropy = agent.train(inp, batch_size=BATCH_SIZE)
                else:
                    loss_p, loss_v, kl, entropy = agent.train(inp, batch_size=BATCH_SIZE, only_value=True)

                avg_pol_loss_list.append(loss_p)
                avg_val_loss_list.append(loss_v)
                avg_true_return_list.append(np.sum(true_rewards)/episodes)
                avg_return_list.append(np.sum(np.concatenate(rewards))/episodes)
                avg_agent_acc_list.append(a_acc)
                avg_expert_acc_list.append(e_acc)

                writer.add_scalar('train/mean_true_return', np.mean(avg_true_return_list), iter)
                writer.add_scalar('train/mean_agent_accuracy', np.mean(avg_agent_acc_list), iter)
                writer.add_scalar('train/mean_expert_accuracy', np.mean(avg_expert_acc_list), iter)
                writer.add_scalar('train/dist', dist, iter)
                # graph.update(np.mean(avg_true_return_list), kl, np.mean(avg_agent_acc_list), np.mean(avg_expert_acc_list), dist)

                if (iter+1)%print_period==0:
                    print('[{}/{}]true return: {:.3f}, return : {:.3f}, value loss : {:.3f}, policy loss : {:.3f}, policy kl : {:.5f}, policy entropy : {:.3f}, a_acc : {:.3f}, e_acc : {:.3f}'.\
                        format(iter, iters, np.mean(avg_true_return_list), np.mean(avg_return_list), np.mean(avg_val_loss_list), np.mean(avg_pol_loss_list), kl, entropy, np.mean(avg_agent_acc_list), np.mean(avg_expert_acc_list)))
                    print("training 하기 :",time.time() - start_t)

                if is_record and iter >= start_train:
                    logger.write(1, [np.mean(avg_true_return_list), np.mean(avg_agent_acc_list), np.mean(avg_expert_acc_list), kl, dist, a_loss, e_loss, gp_loss, np.mean(avg_val_loss_list), np.mean(avg_pol_loss_list)])
                    logger.save()

    if is_root: 
        # graph.update(0, 0, 0, 0, 0, finished=True)
        print("Finish.")
    

if __name__ == "__main__":
    main(is_train=True, is_record=True)

