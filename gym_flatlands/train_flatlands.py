"""
For generating a simple control model for flatlands
"""

from baselines import deepq
from envs import FlatlandsEnv


def callback(lcl, _glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved

def train_deepq():
    """
    Train a flatlands driver with DeepQ
    """
    env = FlatlandsEnv()
    model = deepq.models.mlp([64])
    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-3,
        max_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,
        callback=callback)
    print("Saving model to flatlands.pkl")
    act.save("flatlands.pkl")


if __name__ == '__main__':
    train_deepq()
