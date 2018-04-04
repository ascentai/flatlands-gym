"""
For generating a simple control model for flatlands
"""
import sys
sys.path.append("..")
from baselines.baselines.ppo1 import pposgd_simple


# from mpi4py import MPI
# from baselines.common import set_global_seeds
# from baselines import bench
# import os.path as osp
# from baselines import logger

# from envs import FlatlandsEnv


# def callback(lcl, _glb):
#     # stop training if reward exceeds 199
#     is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
#     return is_solved


def train_bjarne(self, timestamp, **kwargs):
    hid_size = kwargs['hid_size']
    num_hid_layers = kwargs['num_hid_layers']
    max_timesteps = kwargs['max_timesteps']
    timesteps_per_batch = kwargs['timesteps_per_batch']
    clip_param = kwargs['clip_param']
    entcoeff = kwargs['entcoeff']
    optim_epochs = kwargs['optim_epochs']
    optim_stepsize = kwargs['optim_stepsize']
    optim_batchsize = kwargs['optim_batchsize']
    gamma = kwargs['gamma']
    lam = kwargs['lam']
    schedule = kwargs['schedule']
    hyparam_search = kwargs['hyparam_search']
    current_num = kwargs['current_num']
    logging = False
    num_cpu = 8
    seed = 1
    whoami = mpi_fork(num_cpu)
    if whoami == "parent": return
    rank = MPI.COMM_WORLD.Get_rank()
    sess = U.single_threaded_session()
    sess.__enter__()
    logger.session().__enter__()
    if rank != 0: logger.set_level(logger.DISABLED)
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)

    print(current_num)

    def policy_func(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(
            name=name + current_num,
            ob_space=ob_space,
            ac_space=ac_space,
            hid_size=hid_size,
            num_hid_layers=num_hid_layers)

    learned_params, final_res = pposgd_simple.learn(
        sess,
        self.rl_env,
        policy_func,
        max_timesteps=max_timesteps,
        timesteps_per_batch=timesteps_per_batch,
        clip_param=clip_param,
        entcoeff=entcoeff,
        optim_epochs=optim_epochs,
        optim_stepsize=optim_stepsize,
        optim_batchsize=optim_batchsize,
        gamma=gamma,
        lam=lam,
        schedule=schedule,
        logging=logging)
    if not hyparam_search:
        if rank == 0:
            # self.model_path = self.model_path + timestamp + ".pkl"
            # print("Saving model to {}".format(self.model_path))
            pposgd_simple.save(sess, self.hyperparameters, learned_params, "flatlands.pkl")
    return final_res


# def train_cartpole():
#     env = gym.make("flatlands-v0")
#     model = deepq.models.mlp([64])
#     act = deepq.learn(
#         env,
#         q_func=model,
#         lr=1e-3,
#         max_timesteps=100000,
#         buffer_size=50000,
#         exploration_fraction=0.1,
#         exploration_final_eps=0.02,
#         print_freq=10,
#         callback=callback)
#     print("Saving model to flatlands.pkl")
#     act.save("flatlands.pkl")


# def train_atari(env_id, num_timesteps, seed):
#     from baselines.ppo1 import pposgd_simple, cnn_policy
#     import baselines.common.tf_util as U
#     rank = MPI.COMM_WORLD.Get_rank()
#     sess = U.single_threaded_session()
#     sess.__enter__()
#     if rank == 0:
#         logger.configure()
#     else:
#         logger.configure(format_strs=[])
#     env = FlatlandsEnv()

#     def policy_fn(name, ob_space, ac_space):  #pylint: disable=W0613
#         return cnn_policy.CnnPolicy(name=name, ob_space=ob_space, ac_space=ac_space)

#     env = bench.Monitor(env, logger.get_dir() and osp.join(logger.get_dir(), str(rank)))
#     env.seed(workerseed)

#     pposgd_simple.learn(
#         env,
#         policy_fn,
#         max_timesteps=int(num_timesteps * 1.1),
#         timesteps_per_actorbatch=256,
#         clip_param=0.2,
#         entcoeff=0.01,
#         optim_epochs=4,
#         optim_stepsize=1e-3,
#         optim_batchsize=64,
#         gamma=0.99,
#         lam=0.95,
#         schedule='linear')
#     env.close()


def main():
    train_njarne()
    # (env_id="flatlands-v0", num_timesteps=None)


if __name__ == '__main__':
    main()