'''
from gym.envs.registration import register

import os
import os.path as osp
import subprocess

os.environ['QT_PLUGIN_PATH'] = osp.join(osp.dirname(osp.abspath(__file__)), '.qt_plugins') + ':' + \
                               os.environ.get('QT_PLUGIN_PATH','')

register(
    id='DobroHalfCheetah-v1',
    entry_point='dobroEnv:HalfCheetah',
    max_episode_steps=1000,
    reward_threshold=3000.0,
    tags={ "pg_complexity": 8*1000000 },
    )

from dobroEnv.env import HalfCheetah
'''
