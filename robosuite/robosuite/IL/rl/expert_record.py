import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/home/scarab5/hogun_codes/carla-dobro/client')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import dobroEnv
import gym

from nets.ppo import Agent
from dobroEnv.util import run_episodes, random_seed

from pybullet_utils.arg_parser import ArgParser

from mpi4py import MPI
import numpy as np
import random
import pickle
import os
from copy import deepcopy
import time

from client.env import CarlaEnv
from client.observation_utils import CarlaObservationConverter
from client.action_utils import CarlaActionsConverter

ROOT_RANK = 0

def main():
    RANK = MPI.COMM_WORLD.Get_rank()
    NUM_PROC = MPI.COMM_WORLD.Get_size()
    is_root = RANK == ROOT_RANK
    is_train = False
    iters = 1
    episode_count = 10
    max_step = 1

    if is_root:
        arg_parser = ArgParser()
        assert arg_parser.load_file('args.txt')
        obs_converter = CarlaObservationConverter(h=84, w=84, rel_coord_system=False)
        action_converter = CarlaActionsConverter('continuous')
        env = CarlaEnv(obs_converter, action_converter, 0, reward_class_name='CarlaReward', port=2000)
        #env_name = arg_parser.parse_string('env_name')
        game_title = arg_parser.parse_string('game_title')
        save_name = arg_parser.parse_string('save_name')
        #env = gym.make(env_name)
        agent = Agent(env, arg_parser, RANK)

        print("\n###########################")
        print("{} Experiment Start. # of processors : {}".format(game_title, NUM_PROC))

        episodes = 0
        trajectories = []
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
