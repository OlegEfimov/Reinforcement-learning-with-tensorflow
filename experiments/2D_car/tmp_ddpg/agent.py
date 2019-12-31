import numpy as np
from alg_ddpg import DDPG

class Agent(object):
    def __init__(self, config_env):
        self.env = config_env
        self.STATE_DIM = (self.env.STATE_DIM,)
        self.ACTION_DIM = self.env.ACTION_DIM
        self.ACTION_RANGE = np.array(self.env.ACTION_RANGE)
        self.old_state = None
        self.new_state = None
        self.reward = None
        self.action = 0
        self.consecutive_frames = 1
        self.alg = DDPG(self.ACTION_DIM, self.STATE_DIM, self.ACTION_RANGE, self.consecutive_frames)

    def init(self):
        print("Agent init()")


    def handle_new_state(self, arg_str):
        # print("agent-- handle_new_state(arg_str) arg_str=%s" % arg_str)
        args_str = arg_str.split(',')
        state_str = args_str[:self.STATE_DIM[0]]
        reward_str = args_str[self.STATE_DIM[0]]
        terminal_str = args_str[-1]
        if terminal_str == "0":
            done = False
        else:
            done = True

        arr_state_str = np.array(state_str)
        arr_state_float = arr_state_str.astype(np.float)
        new_state_ = arr_state_float
        # self.tmp1 = np.array(np.random.uniform(low=-0.05, high=0.05, size=(6,)))
        # self.old_state = np.array(self.state_)
        self.new_state = np.expand_dims(new_state_, axis=0)

        reward_float = float(reward_str)

        self.reward = reward_float
        if self.old_state is None:
            self.old_state = self.new_state
        # print("state_str=%s" % state_str)
        # print("reward_str=%s" % reward_str)

        self.alg.learn(self.old_state, self.action, self.reward, self.new_state, done)

        self.old_state = self.new_state


        self.action = self.alg.policy_action(self.new_state)

        return self.action


    def handle_save(self, arg_str):
        # self.alg.save()
        print("\nSave Model %s\n")

    def handle_load(self, arg_str):
        # self.alg.load()
        print("\nLoad Model %s\n")
