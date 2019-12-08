import tensorflow as tf
import numpy as np
from config import config


np.random.seed(1)
tf.set_random_seed(1)

LR_A = 1e-4  # learning rate for actor
LR_C = 1e-4  # learning rate for critic
GAMMA = 0.9  # reward discount
REPLACE_ITER_A = 800
REPLACE_ITER_C = 700
MEMORY_CAPACITY = 2000
BATCH_SIZE = 16
VAR_MIN = 0.1
DISCRETE_ACTION = False

env = CarEnv(discrete_action=DISCRETE_ACTION)
STATE_DIM = env.state_dim
ACTION_DIM = env.action_dim
ACTION_BOUND = env.action_bound

# all placeholder for tf
with tf.name_scope('S'):
    S = tf.placeholder(tf.float32, shape=[None, STATE_DIM], name='s')
with tf.name_scope('R'):
    R = tf.placeholder(tf.float32, [None, 1], name='r')
with tf.name_scope('S_'):
    S_ = tf.placeholder(tf.float32, shape=[None, STATE_DIM], name='s_')

MEMORY_CAPACITY = 2000
STATE_DIM = config.state_dim
ACTION_DIM = config.action_dim
ACTION_BOUND = config.action_bound

class Memory(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.capacity  # replace the old memory with new memory
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, n):
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]


class DDPG(object):
    def __init__(self, options):
      self.sess = tf.Session()
      self.M = Memory(MEMORY_CAPACITY, dims=2 * STATE_DIM + ACTION_DIM)

      self.actor = Actor(self.sess, ACTION_DIM, ACTION_BOUND[1], LR_A, REPLACE_ITER_A)
      self.critic = Critic(self.sess, STATE_DIM, ACTION_DIM, LR_C, GAMMA, REPLACE_ITER_C, self.actor.a, self.actor.a_)
      self.actor.add_grad_to_graph(self.critic.a_grads)

    def choose_action(self, s):
        s = s[np.newaxis, :]    # single state
        return self.sess.run(self.a, feed_dict={S: s})[0]  # single action
