"""
Responsible for adding our environment to the Gym registry
Based off of https://github.com/openai/gym/blob/master/gym/envs/__init__.py
"""
from gym.envs.registration import register
# from flatlands import envs

def test_env():
    return "Hello python interpriter"

register(
    id='Flatlands-v0',
    entry_point='flatlands.envs:FlatlandsEnv',
    reward_threshold=1000
)
