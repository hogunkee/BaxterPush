import tensorflow as tf
import numpy as np
import random
import time

def random_seed(seed):
    np.random.seed(seed)
    tf.set_random_seed(seed)
    random.seed(seed)


def get_reward(r_t):
    r_t = r_t[1] + r_t[2] #at mujoco env, and it did very well
    return r_t


def run_episodes(env, agent, episode, is_train, is_root, maxstep, logger=None, is_record=None, pos='random', weather='random'):
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

        obs = env.reset(pos=pos, weather=weather)
        s_t = obs['img']
        # s_t = env.reset()
        while True :
            a_t, value = agent.get_action(s_t, is_train)
            obs1, r_t, done, info = env.step(a_t)
            s_t1 = obs1['img']
            #r_t = get_reward(r_t)

            # if (not is_train) and is_root:
            #     env.render()
            #     time.sleep(4.0) #0.01
            states.append(s_t)
            actions.append(a_t)
            values.append(value)
            true_rewards.append(r_t)
            step += 1
            if done or step >= maxstep:
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


def run_episodes2(env, agent, episode, is_train, is_root, maxstep, eta, logger=None, is_record=None, pos='random', weather='random'):
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
        disc_values = []
        const_values = []
        true_rewards = []

        obs = env.reset(pos=pos, weather=weather)
        s_t = obs['img']
        # s_t = env.reset()
        while True :
            a_t, disc_value, const_value = agent.get_action(s_t, is_train)
            obs1, r_t, done, info = env.step(a_t)
            s_t1 = obs1['img']
            #r_t = get_reward(r_t)

            # if (not is_train) and is_root:
            #     env.render()
            #     time.sleep(4.0)
            states.append(s_t)
            actions.append(a_t)
            disc_values.append(disc_value)
            const_values.append(const_value)
            true_rewards.append(r_t)
            step += 1
            if done or step >= maxstep:
                break
            s_t = s_t1

        if (not is_train) and is_root: 
            print(np.sum(true_rewards))

        next_disc_values = disc_values[1:]
        next_const_values = const_values[1:]
        next_disc_values.append(0)
        next_const_values.append(0)
        states = np.array(states).astype(dtype=np.float32)
        actions = np.array(actions).astype(dtype=np.float32)
        disc_values = np.array(disc_values).astype(dtype=np.float32)
        const_values = np.array(const_values).astype(dtype=np.float32)
        next_disc_values = np.array(next_disc_values).astype(dtype=np.float32)
        next_const_values = np.array(next_const_values).astype(dtype=np.float32)
        trajectories.append([states, actions, disc_values, next_disc_values, const_values, next_const_values, true_rewards])

        if is_record and logger != None:
            logger.write(2, [len(states), np.sum(true_rewards), eta])

    return episodes, trajectories
