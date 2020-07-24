import sys
import os
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '../client'))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nets.ppo import Agent
from dobroEnv.util import run_episodes, random_seed
from dobroEnv.arg_parser import ArgParser

from collections import deque
from copy import deepcopy
import tensorflow as tf
from mpi4py import MPI
import numpy as np
import pickle
import time

from client.env import CarlaEnv
from client.observation_utils import CarlaObservationConverter
from client.action_utils import CarlaActionsConverter

ROOT_RANK = 0


def main():
    RANK = MPI.COMM_WORLD.Get_rank()
    NUM_PROC = MPI.COMM_WORLD.Get_size()
    is_root = RANK == ROOT_RANK

    seed = 1337
    random_seed(seed=seed + RANK)  # for random seed
    now = datetime.datetime.now()
    print(now)

    ################################ configuration #############################################
    arg_parser = ArgParser()
    assert arg_parser.load_file(os.path.join(FILE_PATH, 'args_test.txt'))
    obs_converter = CarlaObservationConverter(h=84, w=84, rel_coord_system=False)
    action_converter = CarlaActionsConverter('continuous')
    env = CarlaEnv(obs_converter, action_converter, 0, reward_class_name='CarlaReward', port=2000, subset='keep_lane', \
                   video_every=1, video_dir=os.path.join(FILE_PATH, 'video', now.strftime("%m%d_%H%M%S"), 'test/'))
    agent = Agent(env, arg_parser, RANK)

    maxstep = arg_parser.parse_int('max_step') // NUM_PROC  # parallel computing
    iters = 1 #arg_parser.parse_int('total_iters')
    game_title = arg_parser.parse_string('game_title')

    if is_root:
        print("\n###########################")
        print("{} Experiment Start. # of processors : {}".format(game_title, NUM_PROC))

    episode = 0
    for iter in range(1):
        episodes, trajectories = run_episodes(env, agent, episode, False, is_root, maxstep, pos=12, weather=0)
        episode += episodes


if __name__ == "__main__":
    main()