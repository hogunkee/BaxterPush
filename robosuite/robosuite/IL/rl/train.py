import sys
import os

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '..'))
sys.path.append(os.path.join(FILE_PATH, '../client'))
# sys.path.append('/home/scarab5/hogun_codes/carla-dobro')
# sys.path.append('/home/scarab5/hogun_codes/carla-dobro/client')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dobroEnv
import gym

from nets.ppo import Agent
from graph_drawer import Graph
from dobroEnv.util import run_episodes, random_seed
from dobroEnv.arg_parser import ArgParser

from collections import deque
from copy import deepcopy
import tensorflow as tf
from mpi4py import MPI
import numpy as np
import pickle
import random
import math
import time

from client.env import CarlaEnv
from client.observation_utils import CarlaObservationConverter
from client.action_utils import CarlaActionsConverter

ROOT_RANK = 0

def main(is_train=True, is_record=True):
    RANK = MPI.COMM_WORLD.Get_rank()
    NUM_PROC = MPI.COMM_WORLD.Get_size()
    is_root = RANK == ROOT_RANK

    if not is_root:
        is_record = False

    if is_record:
        r_records = []
        r_records2 = []

    seed = 1337
    random_seed(seed=seed + RANK) #for random seed

    ################################ configuration #############################################
    arg_parser = ArgParser()
    assert arg_parser.load_file(os.path.join(FILE_PATH, 'args_rungail.txt'))
    # assert arg_parser.load_file('/home/scarab5/hogun_codes/carla-dobro/rl/args_rungail.txt')
    # assert arg_parser.load_file('/home/scarab5/hogun_codes/carla-dobro/rl/args_dobro.txt')
    # env_name = arg_parser.parse_string('env_name')
    # env = gym.make(env_name)
    obs_converter = CarlaObservationConverter(h=84, w=84, rel_coord_system=False)
    action_converter = CarlaActionsConverter('continuous')
    env = CarlaEnv(obs_converter, action_converter, 0, reward_class_name='CarlaReward', port=2000)
    agent = Agent(env, arg_parser, RANK)

    maxstep = arg_parser.parse_int('max_step') // NUM_PROC #parallel computing
    iters = arg_parser.parse_int('total_iters')
    print_period = arg_parser.parse_int('print_period')
    game_title = arg_parser.parse_string('game_title')
    start_train = arg_parser.parse_int('start_train')
    BATCH_SIZE = arg_parser.parse_int('batch_size')
    save_name = arg_parser.parse_string('save_name')
    max_reward = -10000000

    save_name = '.'
    if not os.path.isdir(save_name+'/log'):
        os.makedirs(save_name+'/log')
    if not os.path.isdir(save_name+'/log2'):
        os.makedirs(save_name+'/log2')
    now = time.localtime()
    record_file_name = save_name+"/log/record_%04d%02d%02d_%02d%02d%02d.pkl" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
    record_file_name2 = save_name+"/log2/record_%04d%02d%02d_%02d%02d%02d.pkl" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
    if is_train and is_root: 
        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print("total parameter's num : {}".format(total_parameters))
        graph = Graph(freq=1000, title=game_title.upper(), label=save_name)
    ################################ configuration #############################################


    avg_true_return_list = deque(maxlen=print_period)
    avg_return_list = deque(maxlen=print_period)
    avg_pol_loss_list = deque(maxlen=print_period)
    avg_val_loss_list = deque(maxlen=print_period)

    if is_root:
        print("\n###########################")
        print("{} Experiment Start. # of processors : {}".format(game_title, NUM_PROC))

    episode = 0
    for iter in range(iters):
        start_t = time.time()
        episodes, trajectories = run_episodes(env, agent, episode, is_train, is_root, maxstep)
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

            targets = []
            gaes = []
            score = 0
            for t in trajectories:
                rewards = deepcopy(t[4])
                gae = agent.get_gaes(rewards, t[3], t[2])
                target = np.zeros_like(rewards)
                ret = 0
                for k in reversed(range(len(rewards))):
                    ret = rewards[k] + agent.discount_factor*ret
                    target[k] = ret
                gae = np.array(gae).astype(dtype=np.float32)
                target = np.array(target).astype(dtype=np.float32)
                gaes.append(gae)
                targets.append(target)
                score += np.sum(rewards)
                if is_record:
                    r_records2.append([len(t[0]), np.sum(t[4])])
            gaes = np.concatenate(gaes)
            targets = np.concatenate(targets)
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-6)

            #inp = [states, actions, gaes, targets, goals]
            inps = []
            for proc in range(NUM_PROC):
                if proc == RANK:
                    inp = [states, actions, gaes, targets]
                else:
                    inp = None
                inp = MPI.COMM_WORLD.bcast(inp, root=proc)
                inps.append(deepcopy(inp))
            states = np.concatenate([inp[0] for inp in inps])
            actions = np.concatenate([inp[1] for inp in inps])
            gaes = np.concatenate([inp[2] for inp in inps])
            targets = np.concatenate([inp[3] for inp in inps])
            inp = [states, actions, gaes, targets]

            if is_root:
                if iter >= start_train:
                    loss_p, loss_v, kl, entropy = agent.train(inp, batch_size=BATCH_SIZE)
                else:
                    loss_p, loss_v, kl, entropy = agent.train(inp, batch_size=BATCH_SIZE, only_value=True)
                avg_pol_loss_list.append(loss_p)
                avg_val_loss_list.append(loss_v)
                avg_true_return_list.append(np.sum(true_rewards)/episodes)
                avg_return_list.append(score/episodes)
            agent.sync()
            if is_root:
                #draw graph
                graph.update(np.sum(true_rewards)/episodes, np.mean(avg_val_loss_list), kl, entropy)

                if (iter+1)%print_period==0:
                    print('[{}/{}]return : {:.3f}, value loss : {:.3f}, policy loss : {:.3f}, policy kl : {:.5f}, policy entropy : {:.3f}'.\
                        format(iter, iters, np.mean(avg_return_list), np.mean(avg_val_loss_list), np.mean(avg_pol_loss_list), kl, entropy))
                    print("training 하기 :",time.time() - start_t)

                if is_record and iter >= start_train:
                    r_records.append([np.mean(avg_true_return_list), np.mean(avg_pol_loss_list), kl])
                    with open(record_file_name, 'wb') as f:
                        pickle.dump(r_records, f)
                    with open(record_file_name2, 'wb') as f:
                        pickle.dump(r_records2, f)

    if is_root: 
        graph.update(0, 0, 0, 0, finished=True)
        print("Finish.")
    

if __name__ == "__main__":
    main(is_train=True, is_record=True)
