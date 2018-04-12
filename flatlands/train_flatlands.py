"""
Train PPO in the Flatlands Gym environment.
"""

import sys
import os
import argparse

from mpi4py import MPI
import gym
from baselines import bench
from baselines import logger
import baselines.common.tf_util as tfu

import flatlands
import pposgd_flatlands
from flatlands_policy import flatPolicy


def _policy_fn(name, ob_space, ac_space):
    """
    Instantiates a `flatPolicy` as PPO's optimizer.

    Inputs: name        desired identifier
            ob_space    placeholder representing the dimensions of the
                            environment's observation space
            ac_space    placeholder representing the dimensions of the
                            environment's action space

    Return: an instance of `flatPolicy`
    """
    return flatPolicy(name=name, ob_space=ob_space, ac_space=ac_space)

def train(timesteps):
    """
    Kicks off the training routine for PPO.

    Inputs: timesteps   desired length of training

    Return: None; the trained model will be saved according to
        `pposgd_flatlands.learn()`
    """
    rank = MPI.COMM_WORLD.Get_rank()
    session = tfu.single_threaded_session()
    session.__enter__()
    if rank == 0:
        logger.configure()
    else:
        logger.configure(format_strs=[])

    env = gym.make("Flatlands-v0")
    env = bench.Monitor(env, logger.get_dir() and
            os.path.join(logger.get_dir(), str(rank)))

    pposgd_flatlands.learn(env, _policy_fn,
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
    Passes commandline arguments to `train()`.
    """
    parser = argparse.ArgumentParser(description="Train PPOSGD in Flatlands.")
    parser.add_argument("--steps", type=int, default=int(10e6))

    args = parser.parse_args()
    train(args.steps)

if __name__ == "__main__":
    main()
