import gym
import numpy as np
import os
from itertools import count
import matplotlib.pyplot as plt
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.sac.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import SAC
seed_num=2019
env_name = 'HalfCheetah'
env = gym.make('HalfCheetah-v2')

log_dir = "./GAIL_HalfCheetah"
os.makedirs(log_dir, exist_ok=True)
model = SAC(MlpPolicy, env, verbose=1)
env = Monitor(env, log_dir, allow_early_resets=True)
model.learn(total_timesteps=1500000, log_interval=100)
def moving_average(values, window):
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')

def plot_results(log_folder, title='Learning Curve'):
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.show()
plot_results(log_dir)



"""
for i_episode in count():
    obs = env.reset()
    dones = False
    reward = 0
    
    while not dones:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        reward+=rewards
        expert_traj.append(np.concatenate((obs, action), axis =1))
        num_steps += 1
    
    print("episode:", i_episode, "reward:", reward)
    
    if num_steps >= max_expert_num:
        break

a, b, c= np.shape(expert_traj)
expert_traj = np.reshape(expert_traj, (a, c))
print()
print(np.shape(expert_traj))
print()
filename="expert_traj_" + str(seed_num)+"_"+ env_name +".npy"
np.save(filename, expert_traj)

env.close()
"""
