"""
Evaluate a trained PPO model in Flatlands.
"""

import argparse

import gym
import baselines.common.tf_util as tfu
from baselines import logger

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

def evaluate(model_path):
    """
    Sets up the given PPO model in the Flatlands environment.

    Inputs: model_path  the path to the directory containing the desired model
                            as a saved TensorFlow session

    Return: None; the environment will be rendered until interrupted, printing
        the reward for each episode
    """
    session = tfu.single_threaded_session()
    session.__enter__()

    env = gym.make("flatlands-v0")
    obs = env.reset()

    pposgd_flatlands.load(model_path, env, _policy_fn,
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

    while True:
        obs = env.reset()
        done = False
        ep_rew = 0
        while not done:
            env.render()
            action = pposgd_flatlands.act(True, obs)
            obs, rew, done, info = env.step(action[0])
            ep_rew += rew

        print("Episode reward: %i" % ep_rew)

def main():
    """
    Passes commandline arguments to `evaluate()`.
    """
    parser = argparse.ArgumentParser(description=("Evaluate PPOSGD in"
            + " Flatlands."))
    parser.add_argument("model_path", help="path to the saved TF session")
    args = parser.parse_args()
    
    evaluate(args.model_path)

if __name__ == "__main__":
    main()
