"""
Responsible for adding our environment to the Gym registry
Based off of https://github.com/openai/gym/blob/master/gym/envs/__init__.py
"""
from gym.envs.registration import register

register(
    id='flatlands-v0',
    entry_point='gym_flatlands.envs:FlatlandsEnv',
    reward_threshold=1000
)
