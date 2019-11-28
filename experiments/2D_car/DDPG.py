"""
Environment is a 2D car.
Car has 5 sensors to obtain distance information.

Car collision => reward = -1, otherwise => reward = 0.
 
You can train this RL by using LOAD = False, after training, this model will be store in the a local folder.
Using LOAD = True to reload the trained model for playing.

You can customize this script in a way you want.

View more on [莫烦Python] : https://morvanzhou.github.io/tutorials/

Requirement:
pyglet >= 1.2.4
numpy >= 1.12.1
tensorflow >= 1.0.1
"""

import tensorflow as tf
import numpy as np
import os
import shutil
from remote_car_env import RemoteCarEnv

TRAIN_LOOP = {"state": "start"}
USERS = set()



np.random.seed(1)
tf.set_random_seed(1)

MAX_EPISODES = 500
MAX_EP_STEPS = 600
LR_A = 1e-4  # learning rate for actor
LR_C = 1e-4  # learning rate for critic
GAMMA = 0.9  # reward discount
REPLACE_ITER_A = 800
REPLACE_ITER_C = 700
MEMORY_CAPACITY = 2000
BATCH_SIZE = 16
VAR_MIN = 0.1
RENDER = True
LOAD = False
DISCRETE_ACTION = False

remoteEnv = RemoteCarEnv()
STATE_DIM = remoteEnv.state_dim
ACTION_DIM = remoteEnv.action_dim
ACTION_BOUND = remoteEnv.action_bound

# all placeholder for tf
with tf.name_scope('S'):
    S = tf.placeholder(tf.float32, shape=[None, STATE_DIM], name='s')
with tf.name_scope('R'):
    R = tf.placeholder(tf.float32, [None, 1], name='r')
with tf.name_scope('S_'):
    S_ = tf.placeholder(tf.float32, shape=[None, STATE_DIM], name='s_')


class Actor(object):
    def __init__(self, sess, action_dim, action_bound, learning_rate, t_replace_iter):
        self.sess = sess
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.lr = learning_rate
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0

        with tf.variable_scope('Actor'):
            # input s, output a
            self.a = self._build_net(S, scope='eval_net', trainable=True)

            # input s_, output a, get a_ for critic
            self.a_ = self._build_net(S_, scope='target_net', trainable=False)

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')

    def _build_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.contrib.layers.xavier_initializer()
            init_b = tf.constant_initializer(0.001)
            net = tf.layers.dense(s, 100, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l1',
                                  trainable=trainable)
            net = tf.layers.dense(net, 20, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l2',
                                  trainable=trainable)
            with tf.variable_scope('a'):
                actions = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, kernel_initializer=init_w,
                                          name='a', trainable=trainable)
                scaled_a = tf.multiply(actions, self.action_bound, name='scaled_a')  # Scale output to -action_bound to action_bound
        return scaled_a

    def learn(self, s):   # batch update
        self.sess.run(self.train_op, feed_dict={S: s})
        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])
        self.t_replace_counter += 1

    def choose_action(self, s):
        s = s[np.newaxis, :]    # single state
        return self.sess.run(self.a, feed_dict={S: s})[0]  # single action

    def add_grad_to_graph(self, a_grads):
        with tf.variable_scope('policy_grads'):
            self.policy_grads = tf.gradients(ys=self.a, xs=self.e_params, grad_ys=a_grads)

        with tf.variable_scope('A_train'):
            opt = tf.train.RMSPropOptimizer(-self.lr)  # (- learning rate) for ascent policy
            self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))


class Critic(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, gamma, t_replace_iter, a, a_):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0

        with tf.variable_scope('Critic'):
            # Input (s, a), output q
            self.a = a
            self.q = self._build_net(S, self.a, 'eval_net', trainable=True)

            # Input (s_, a_), output q_ for q_target
            self.q_ = self._build_net(S_, a_, 'target_net', trainable=False)    # target_q is based on a_ from Actor's target_net

            self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_net')
            self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')

        with tf.variable_scope('target_q'):
            self.target_q = R + self.gamma * self.q_

        with tf.variable_scope('TD_error'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q))

        with tf.variable_scope('C_train'):
            self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        with tf.variable_scope('a_grad'):
            self.a_grads = tf.gradients(self.q, a)[0]   # tensor of gradients of each sample (None, a_dim)

    def _build_net(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.contrib.layers.xavier_initializer()
            init_b = tf.constant_initializer(0.01)

            with tf.variable_scope('l1'):
                n_l1 = 100
                w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], initializer=init_w, trainable=trainable)
                w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=init_w, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
                net = tf.nn.relu6(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net = tf.layers.dense(net, 20, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l2',
                                  trainable=trainable)
            with tf.variable_scope('q'):
                q = tf.layers.dense(net, 1, kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)   # Q(s,a)
        return q

    def learn(self, s, a, r, s_):
        self.sess.run(self.train_op, feed_dict={S: s, self.a: a, R: r, S_: s_})
        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])
        self.t_replace_counter += 1


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

    def initialSample(self, n):
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]


sess = tf.Session()

# Create actor and critic.
actor = Actor(sess, ACTION_DIM, ACTION_BOUND[1], LR_A, REPLACE_ITER_A)
critic = Critic(sess, STATE_DIM, ACTION_DIM, LR_C, GAMMA, REPLACE_ITER_C, actor.a, actor.a_)
actor.add_grad_to_graph(critic.a_grads)

M = Memory(MEMORY_CAPACITY, dims=2 * STATE_DIM + ACTION_DIM + 1)

saver = tf.train.Saver()
path = './discrete' if DISCRETE_ACTION else './continuous'


# print("step_counter = %s" % str(step_counter))

if LOAD:
    saver.restore(sess, tf.train.latest_checkpoint(path))
else:
    sess.run(tf.global_variables_initializer())

var = 2.  # control exploration
ep_counter = 0
step_counter = 0
s_ = None
r = None
done = None
s = None
s_ = None
a = None
done = False
b_M = None
b_s = None
b_a = None
b_r = None
b_s_ = None


def state_selector(arg): 
    switcher = { 
        "start": start_handler, 
        "wait_init_done": wait_init_done_handler,
        "start_episode": start_episode_handler,
        "stop_episode": stop_episode_handler,
        "send_reset": send_reset_handler,
        "wait_reset_done": wait_reset_done_handler,
        "start_step": start_step_handler,
        "stop_step": stop_step_handler,
        "nn_choose_act": nn_choose_act_handler,
        "env_step": env_step_handler,
        "wait_step_done": wait_step_done_handler,
        "nn_learn": nn_learn_handler,
        "stop": stop_handler
    } 
    return switcher.get(arg, unknown_state_handler)

def start_handler():
    print("DDPG - start_handler")
    remoteEnv.init()
    return "wait_init_done"

def wait_init_done_handler():
    print("DDPG - wait_init_done_handler")
    global s
    global s_
    global r
    global done
    global a
    if remoteEnv.init_done:
        remoteEnv.init_done = False
        s_ = remoteEnv.env_state
        r = remoteEnv.env_reward
        done = remoteEnv.env_done
        a = remoteEnv.sample_action
        return "start_episode"
    else :
        return "wait_init_done"

def start_episode_handler():
    print("DDPG - start_episode_handler")
    global ep_counter
    ep_counter = 0
    return "send_reset"

def send_reset_handler():
    print("DDPG - send_reset_handler")
    global step_counter
    step_counter = 0
    remoteEnv.reset()
    return "wait_reset_done"

def wait_reset_done_handler():
    print("DDPG - wait_reset_done_handler")
    global s
    if remoteEnv.reset_done:
        remoteEnv.reset_done = False
        s = remoteEnv.env_state
        return "start_step"
    else :
        return "wait_reset_done"

def start_step_handler():
    print("DDPG - start_step_handler")
    # remoteEnv.render()
    return "nn_choose_act"

def stop_step_handler():
    print("DDPG - stop_step_handler")
    global step_counter
    # print("---------------------------------step_counter = %s" % str(step_counter))
    step_counter += 1
    if done or step_counter >= MAX_EP_STEPS:
        return "stop_episode"
    else :
        return "start_step"

def stop_episode_handler():
    print("DDPG - stop_episode_handler")
    global ep_counter
    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!ep_counter = %s" % str(ep_counter))
    ep_counter += 1
    if ep_counter < MAX_EPISODES:
        return "send_reset"
    else :
        return "stop"

def stop_handler():
    print("DDPG - stop_handler")
    return "end"

def unknown_state_handler():
    print("DDPG - unknown_state_handler")
    return "end"

def nn_choose_act_handler():
    print("DDPG - nn_choose_act_handler")
    global a
    a = actor.choose_action(s)
    a = np.clip(np.random.normal(a, var), *ACTION_BOUND)    # add randomness to action selection for exploration
    return "env_step"

def env_step_handler():
    print("DDPG - env_step_handler")
    remoteEnv.step(a)
    return "wait_step_done"

def wait_step_done_handler():
    print("DDPG - wait_step_done_handler")
    global s_
    global r
    global done
    if remoteEnv.step_done:
        remoteEnv.step_done = False
        s_ = remoteEnv.env_state
        r = remoteEnv.env_reward
        done = remoteEnv.env_done
        M.store_transition(s, a, r, s_)
        return "nn_learn"
    else :
        return "wait_step_done"


def nn_learn_handler():
    print("DDPG - nn_learn_handler")
    global var
    global b_M
    global b_s
    global b_a
    global b_r
    global b_s_
    global s
    global critic
    global actor

    if M.pointer > MEMORY_CAPACITY:
        var = max([var*.9995, VAR_MIN])    # decay the action randomness
        b_M = M.sample(BATCH_SIZE)
        b_s = b_M[:, :STATE_DIM]
        b_a = b_M[:, STATE_DIM: STATE_DIM + ACTION_DIM]
        b_r = b_M[:, -STATE_DIM - 1: -STATE_DIM]
        b_s_ = b_M[:, -STATE_DIM:]

        critic.learn(b_s, b_a, b_r, b_s_)
        actor.learn(b_s)

    s = s_
    return "stop_step"


def train_loop():
    print("DDPG - train_loop")
    while TRAIN_LOOP["state"] != "end":
        state = TRAIN_LOOP["state"]
        stateHandler = state_selector(state)
        new_state = stateHandler()
        TRAIN_LOOP["state"] = new_state
        # print("%s\t->\t %s" % (state, new_state))

if __name__ == '__main__':
    train_loop()
