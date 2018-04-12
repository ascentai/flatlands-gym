"""
Flatlands optimizer for PPO.
"""

import tensorflow as tf
import baselines.common.tf_util as U
import gym
from baselines.common.distributions import make_pdtype


class flatPolicy(object):
    """
    An optimizer policy for use with PPO in Flatlands.
    """
    def __init__(self, name, ob_space, ac_space, kind="large"):
        self.recurrent = False
        with tf.variable_scope(name):
            self._init(ob_space, ac_space, kind)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, kind):
        assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = make_pdtype(ac_space)
        sequence_length = None

        ob = U.get_placeholder(name="ob", dtype=tf.float32,
                shape=[sequence_length]+list(ob_space.shape))

        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(14, activation="relu",
                    input_shape=ob_space.shape),
            tf.keras.layers.Dense(7, activation="relu"),
            tf.keras.layers.Dense(ac_space.shape[0])
        ])

        logits = tf.layers.dense(ob, self.pdtype.param_shape()[0],
                name="logits", kernel_initializer=U.normc_initializer(0.01))

        self.pd = self.pdtype.pdfromflat(logits)
        self.vpred = tf.layers.dense(ob, 1, name="value",
                kernel_initializer=U.normc_initializer(1.0))[:,0]

        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = self.pd.sample()
        self._act = U.function([stochastic, ob], [ac, self.vpred])

    def act(self, stochastic, ob):
        """
        Generates an action to be executed in the environment.

        Inputs: stochastic  whether to use stochastic approximation of the
                                gradient descent algorithm
                ob          an observation from the environment

        Return: an optimal action and associated predicted values
        """
        ac1, vpred1 = self._act(stochastic, [ob])
        return ac1[0], vpred1[0]

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_initial_state(self):
        return []
