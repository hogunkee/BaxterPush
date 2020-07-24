import sys
import os
#sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

ROOT_RANK = 0

def main():
    RANK = MPI.COMM_WORLD.Get_rank()
    NUM_PROC = MPI.COMM_WORLD.Get_size()
    is_root = RANK == ROOT_RANK
    is_train = False
    step_count = 8000

    if is_root:
        arg_parser = ArgParser()
        assert arg_parser.load_file('args.txt')
        env_name = arg_parser.parse_string('env_name')
        game_title = arg_parser.parse_string('game_title')
        save_name = arg_parser.parse_string('save_name')
        maxstep = arg_parser.parse_int('max_step') // NUM_PROC
        env = gym.make(env_name)
        agent = Agent(env, arg_parser, RANK)

        print("\n###########################")
        print("{} Experiment Start. # of processors : {}".format(game_title, NUM_PROC))

        episodes = 0
        trajectories = []
        while True:
            episode, trajectory = run_episodes(env, agent, episodes, is_train, False, maxstep)
            print(np.sum(trajectory[0][4]))
            trajectories += trajectory
            episodes += episode
            if np.sum([len(t[0]) for t in trajectories]) >= step_count:
                break

        expert_states = np.concatenate([t[0] for t in trajectories])
        expert_actions = np.concatenate([t[1] for t in trajectories])
        with open('../trajectory/fail_s.pkl', 'wb') as f:
            pickle.dump(expert_states, f)
        with open('../trajectory/fail_a.pkl', 'wb') as f:
            pickle.dump(expert_actions, f)
        print("save trajectory!")
        print("Finish.")

if __name__ == "__main__":
    main()

