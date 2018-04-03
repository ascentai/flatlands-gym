"""
TODO
"""

import sys
import os
import argparse

from mpi4py import MPI
import gym
from baselines.common import set_global_seeds
from baselines import bench
from baselines import logger
from baselines.ppo1 import pposgd_simple, cnn_policy
import baselines.common.tf_util as tfu

from envs import FlatlandsEnv
from flatlands_policy import flatPolicy


def _policy_fn(name, ob_space, ac_space):
    """
    TODO
    """
    #return cnn_policy.CnnPolicy(name=name, ob_space=ob_space,
    #        ac_space=ac_space)
    return flatPolicy(name=name, ob_space=ob_space, ac_space=ac_space)

def train(timesteps, seed):
    """
    TODO
    """
    rank = MPI.COMM_WORLD.Get_rank()
    session = tfu.single_threaded_session()
    session.__enter__()
    if rank == 0:
        logger.configure()
    else:
        logger.configure(format_strs=[])

    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)

    env = gym.make("flatlands-v0")
    env.reset()
    env = bench.Monitor(env, logger.get_dir() and
            os.path.join(logger.get_dir(), str(rank)))
    env.seed(workerseed)

    #env = wrap_deepmind(env)
    #env.seed(workerseed)

    pposgd_simple.learn(env, _policy_fn,
            max_timesteps=int(timesteps * 1.1),
            timesteps_per_actorbatch=256,
            clip_param=0.2,
            entcoeff=0.01,
            optim_epochs=4,
            optim_stepsize=1e-3,
            optim_batchsize=64,
            gamma=0.99,
            lam=0.95,
            schedule="linear"
    )
    env.close()

def main():
    """
    TODO
    """
    parser = argparse.ArgumentParser(description="Train PPOSGD in Flatlands.")
    parser.add_argument("--steps", type=int, default=int(10e6))
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    train(args.steps, args.seed)

if __name__ == "__main__":
    main()
