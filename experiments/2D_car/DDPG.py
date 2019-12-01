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
from car_env import CarEnv
# from websocket_server import WebsocketServer
import asyncio
import websockets
USERS = set()


# def new_client(client, server):
    # print("New client connected and was given id %d" % client['id'])
    # server.send_message_to_all("Hey all, a new client has joined us")
    # server.send_message(client, "0,0")


# Called for every client disconnecting
# def client_left(client, server):
    # print("Client(%d) disconnected" % client['id'])


inputNN = []
# Called when a client sends a message
# def message_received(client, server, message):

#     inputNN = message.split(',')
#     for i in range(0, len(inputNN)):
#         inputNN[i] = float(inputNN[i])
#     print("Client(%d) said: %s" % (client['id'], message))
#     # state = self.sensor_info[:, 0].flatten()/self.sensor_max
#     # inputNN_tf = tf.constant(inputNN)

#     # actionArray = calculateAction(inputNN_tf)
#     # server.send_message(client, "0.7,-0.3")


np.random.seed(1)
tf.set_random_seed(1)

MAX_EPISODES = 1500
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
# LOAD = True
LOAD = False
DISCRETE_ACTION = False

env = CarEnv(discrete_action=DISCRETE_ACTION)
STATE_DIM = env.state_dim
ACTION_DIM = env.action_dim
ACTION_BOUND = env.action_bound
state = 'start'

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
        # print("Actor _build_net")
        with tf.variable_scope(scope):
            init_w = tf.contrib.layers.xavier_initializer()
            init_b = tf.constant_initializer(0.001)
            net = tf.layers.dense(s, 60, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l1',
                                  trainable=trainable)
            net = tf.layers.dense(net, 40, activation=tf.nn.relu,
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
        # print("Critic _build_net")
        with tf.variable_scope(scope):
            init_w = tf.contrib.layers.xavier_initializer()
            init_b = tf.constant_initializer(0.01)

            with tf.variable_scope('l1'):
                n_l1 = 60
                w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], initializer=init_w, trainable=trainable)
                w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=init_w, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
                net = tf.nn.relu6(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net = tf.layers.dense(net, 80, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l2',
                                  trainable=trainable)
            net = tf.layers.dense(net, 70, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l3',
                                  trainable=trainable)
            net = tf.layers.dense(net, 60, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l4',
                                  trainable=trainable)
            net = tf.layers.dense(net, 50, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l5',
                                  trainable=trainable)
            with tf.variable_scope('q'):
                q = tf.layers.dense(net, 1, kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)   # Q(s,a)
        return q

    def learn(self, s, a, r, s_):
        self.sess.run(self.train_op, feed_dict={S: s, self.a: a, R: r, S_: s_})
        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])
        self.t_replace_counter += 1

    def getLoss(self, s, a, r, s_):
        return self.sess.run(self.loss, feed_dict={S: s, self.a: a, R: r, S_: s_})


class Memory(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0
        # print("Memory __init__")

    def store_transition(self, s, a, r, s_):
        # print("Memory store_transition")
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.capacity  # replace the old memory with new memory
        self.data[index, :] = transition
        self.pointer += 1
        # print("Memory pointer: %s" % str(self.pointer))

    def sample(self, n):
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        # print("Memory sample")
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

if LOAD:
    saver.restore(sess, tf.train.latest_checkpoint(path))
else:
    sess.run(tf.global_variables_initializer())


# def train():
#     var = 2.  # control exploration
#     for ep in range(MAX_EPISODES):
#         s = env.reset()
#         ep_step = 0

#         for t in range(MAX_EP_STEPS):
#         # while True:
#             if RENDER:
#                 env.render()

#             # Added exploration noise
#             a = actor.choose_action(s)
#             a = np.clip(np.random.normal(a, var), *ACTION_BOUND)    # add randomness to action selection for exploration
#             s_, r, done = env.step(a)
#             M.store_transition(s, a, r, s_)

#             if M.pointer > MEMORY_CAPACITY:
#                 var = max([var*.9995, VAR_MIN])    # decay the action randomness
#                 b_M = M.sample(BATCH_SIZE)
#                 b_s = b_M[:, :STATE_DIM]
#                 b_a = b_M[:, STATE_DIM: STATE_DIM + ACTION_DIM]
#                 b_r = b_M[:, -STATE_DIM - 1: -STATE_DIM]
#                 b_s_ = b_M[:, -STATE_DIM:]

#                 critic.learn(b_s, b_a, b_r, b_s_)
#                 actor.learn(b_s)

#             s = s_
#             ep_step += 1

#             if done or t == MAX_EP_STEPS - 1:
#             # if done:
#                 print('Ep:', ep,
#                       '| Steps: %i' % int(ep_step),
#                       '| Explore: %.2f' % var,
#                       )
#                 break

#     if os.path.isdir(path): shutil.rmtree(path)
#     os.mkdir(path)
#     ckpt_path = os.path.join(path, 'DDPG.ckpt')
#     save_path = saver.save(sess, ckpt_path, write_meta_graph=False)
#     print("\nSave Model %s\n" % save_path)

async def train(websocket):
    await websocket.send("hello")
    response = await websocket.recv()
    # print('-------!!!!!------')
    # print(response)
 
    # var = 2.  # control exploration
    # while state != 'end':
    #     if state == 'need_action':
    #         inputNN_tf = tf.constant(inputNN)
    #         actionArray = calculateAction(inputNN_tf)
    #         action_as_string = actionArray.join(',')
    #         server.send_message(client, action_as_string)

    #     if state == 'need_learn':
    #         critic.learn(b_s, b_a, b_r, b_s_)
    #         actor.learn(b_s)



def calculateAction(stateIn):
    # print("-----calculateAction(stateIn=")
    # print(stateIn)
    var = 0.5  # control exploration
    # Added exploration noise
    # state = env.reset()
    a = actor.choose_action(stateIn)
    tmp333 = np.random.normal(a, var)
    # print(tmp333)
    a = np.clip(np.random.normal(a, var), *ACTION_BOUND)    # add randomness to action selection for exploration
    # print(a)
    return a

def oneStepLearn(state, action, reward, new_state):
    var = 2.  # control exploration
    M.store_transition(state, action, reward, new_state)

    if M.pointer > MEMORY_CAPACITY:
        var = max([var*.9995, VAR_MIN])    # decay the action randomness
        b_M = M.sample(BATCH_SIZE)
        b_s = b_M[:, :STATE_DIM]
        b_a = b_M[:, STATE_DIM: STATE_DIM + ACTION_DIM]
        b_r = b_M[:, -STATE_DIM - 1: -STATE_DIM]
        b_s_ = b_M[:, -STATE_DIM:]

        critic.learn(b_s, b_a, b_r, b_s_)
        actor.learn(b_s)



def eval():
    env.set_fps(30)
    while True:
        s = env.reset()
        while True:
            env.render()
            a = actor.choose_action(s)
            s_, r, done = env.step(a)
            s = s_
            if done:
                break

async def notify_clients(message):
    if USERS:  # asyncio.wait doesn't accept an empty list
        mess = message
        print("send %s" % str(mess))
        await asyncio.wait([user.send(mess) for user in USERS])

async def register(websocket):
    USERS.add(websocket)

async def unregister(websocket):
    USERS.remove(websocket)

def message_received(message):
    inputNN = message.split(',')
    for i in range(0, len(inputNN)):
        inputNN[i] = float(inputNN[i])
    return inputNN

async def counter(websocket, path):
    path2 = './discrete' if DISCRETE_ACTION else './continuous'
    var = 2.  # control exploration
    ep = 0
    ep_step = 0
    sendActionCounter = 0
    receiveStateCounter = 0
    receiveRewardCounter = 0
    await register(websocket)
    s_ = env.reset()
    await notify_clients('reset')
    s = s_
    actionArray = calculateAction(s_)
    try:
        async for message in websocket:
            tmp = message_received(message)
            if len(tmp) > 2:
                print("receive state: %s" % message)
                receiveStateCounter += 1
                # print("receiveStateCounter =  %s" % str(receiveStateCounter))
                state_input = np.array(tmp, dtype=np.float64)
                s_ = state_input
                actionArray = calculateAction(state_input)
                action_as_string = ''
                for num in actionArray:
                    action_as_string += str(num) + ','
                sendActionCounter += 1
                # print("sendActionCounter =  %s" % str(sendActionCounter))
                await notify_clients(action_as_string[:-1])
            else:
                # reward
                tmp_loss = 0
                reward = tmp[0]
                done = tmp[1]
                print("receive reward: %s" % reward)
                receiveRewardCounter += 1
                # print("receiveRewardCounter =  %s" % str(receiveRewardCounter))
                r = reward
                M.store_transition(s, actionArray, r, s_)
                # print("M.pointer: %s" % str(M.pointer))
                if M.pointer > MEMORY_CAPACITY:
                    var = max([var*.9995, VAR_MIN])    # decay the action randomness
                    b_M = M.sample(BATCH_SIZE)
                    b_s = b_M[:, :STATE_DIM]
                    b_a = b_M[:, STATE_DIM: STATE_DIM + ACTION_DIM]
                    b_r = b_M[:, -STATE_DIM - 1: -STATE_DIM]
                    b_s_ = b_M[:, -STATE_DIM:]

                    # print("start learning")
                    tmp_loss = critic.getLoss(b_s, b_a, b_r, b_s_)
                    # print("tmp_loss: %s" % str(tmp_loss))
                    critic.learn(b_s, b_a, b_r, b_s_)
                    actor.learn(b_s)
                    # print("end learning")
                s = s_
                ep_step += 1
                if done or ep_step > MAX_EP_STEPS - 1:
                    if done:
                        print('Ep:', ep,
                              '| Steps: %i' % int(ep_step),
                              '| Explore: %.2f' % var,
                              '| M.pointer: %s' % str(M.pointer)
                              )
                    ep += 1
                    ep_step = 0
                    if ep > MAX_EPISODES - 1:
                        ep = 0
                        await notify_clients('stop')
                        if os.path.isdir(path2): shutil.rmtree(path2)
                        os.mkdir(path2)
                        ckpt_path = os.path.join(path2, 'DDPG.ckpt')
                        save_path = saver.save(sess, ckpt_path, write_meta_graph=False)
                        print("\nSave Model %s\n" % save_path)

                    else:
                        await notify_clients('reset')
                else:
                    await notify_clients('loss:' + str(tmp_loss))
    finally:
        await unregister(websocket)

    # #get new state
    # s_new = state_input
    # a = calculateAction(state_input)
    # notify_clients(action_as_string[:-1])
    # #wait reward
    # #get reward
    # r = reward
    # M.store_transition(s_old, a, r, s_new)
    # if M.pointer > MEMORY_CAPACITY:
    #     var = max([var*.9995, VAR_MIN])    # decay the action randomness
    #     b_M = M.sample(BATCH_SIZE)
    #     b_s = b_M[:, :STATE_DIM]
    #     b_a = b_M[:, STATE_DIM: STATE_DIM + ACTION_DIM]
    #     b_r = b_M[:, -STATE_DIM - 1: -STATE_DIM]
    #     b_s_ = b_M[:, -STATE_DIM:]

    #     critic.learn(b_s, b_a, b_r, b_s_)
    #     actor.learn(b_s)

    # s_old = s_new


if __name__ == '__main__':
    if LOAD:
        # eval()
        start_server = websockets.serve(counter, "localhost", 9001)

        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()
    else:
        # train()
        start_server = websockets.serve(counter, "localhost", 9001)

        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()
