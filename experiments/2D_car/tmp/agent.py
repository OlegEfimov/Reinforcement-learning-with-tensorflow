import tensorflow as tf
import numpy as np

from DDPG2 import Actor
from DDPG2 import Critic
from DDPG2 import Memory

class Agent(object):
    def __init__(self, config_env):
        np.random.seed(1)
        # tf.set_random_seed(1)
        tf.compat.v1.set_random_seed(1)
        self.LR_A = 1e-4  # learning rate for actor
        self.LR_C = 1e-4  # learning rate for critic
        self.GAMMA = 0.9  # reward discount
        self.REPLACE_ITER_A = 800
        self.REPLACE_ITER_C = 700
        self.MEMORY_CAPACITY = 2000
        self.BATCH_SIZE = 16
        self.VAR_MIN = 0.1
        self.DISCRETE_ACTION = False

        self.env = config_env
        self.STATE_DIM = self.env.STATE_DIM
        self.ACTION_DIM = self.env.ACTION_DIM
        self.ACTION_BOUND = self.env.ACTION_BOUND
        self.state = None
        self.state_ = None
        self.reward = None

        self.sess = tf.Session()
        # Create actor and critic.
        self.actor = Actor(self.sess, self.ACTION_DIM, self.ACTION_BOUND[1], self.LR_A, self.REPLACE_ITER_A)
        self.critic = Critic(self.sess, self.STATE_DIM, self.ACTION_DIM, self.LR_C, self.GAMMA, self.REPLACE_ITER_C, self.actor.a, self.actor.a_)
        self.actor.add_grad_to_graph(self.critic.a_grads)

        self.M = Memory(self.MEMORY_CAPACITY, dims=2 * self.STATE_DIM + self.ACTION_DIM + 1)

    def handle_new_state(self, arg_str):
        args_str = arg_str.split(',')
        state_str = args_str[:state_dim]
        reward_str = args_str[state_dim]
        arr_state_str = np.array(state_str)
        arr_state_float = arr_state_str.astype(np.float)
        self.state_ = arr_state_float
        reward_float = float(reward_str)
        self.reward = reward_float
        self.M.store_transition(self.state, self.action, self.reward, self.state_)
        if self.M.pointer > self.MEMORY_CAPACITY:
            self.var = max([self.var*.9995, self.VAR_MIN])    # decay the action randomness
            b_M = self.M.sample(self.BATCH_SIZE)
            b_s = b_M[:, :self.STATE_DIM]
            b_a = b_M[:, self.STATE_DIM: self.STATE_DIM + self.ACTION_DIM]
            b_r = b_M[:, -self.TATE_DIM - 1: -self.STATE_DIM]
            b_s_ = b_M[:, -self.STATE_DIM:]

            self.critic.learn(b_s, b_a, b_r, b_s_)
            self.actor.learn(b_s)

        self.state = self.state_

        action = actor.choose_action(self.state_)
        # add randomness to action selection for exploration
        self.action = np.clip(np.random.normal(action, var), *ACTION_BOUND)
        return self.action


