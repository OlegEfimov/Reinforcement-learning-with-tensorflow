import random
import numpy as np

from tqdm import tqdm
from keras.models import Model
from keras import regularizers
from keras.utils import to_categorical
from keras.layers import Input, Dense, Flatten

from critic import Critic
from actor import Actor
# from utils.networks import tfSummary
# from utils.stats import gather_stats

class A2C:
    """ Actor-Critic Main Algorithm
    """

    def __init__(self, act_dim, env_dim, k, gamma = 0.99, lr = 0.0001):
        """ Initialization
        """
        # Environment and A2C parameters
        self.act_dim = act_dim
        self.env_dim = (k,) + (env_dim,)
        self.gamma = gamma
        self.lr = lr
        # Create actor and critic networks
        self.shared = self.buildNetwork()
        self.actor = Actor(self.env_dim, act_dim, self.shared, lr)
        self.critic = Critic(self.env_dim, act_dim, self.shared, lr)
        # Build optimizers
        self.a_opt = self.actor.optimizer()
        self.c_opt = self.critic.optimizer()
        # self.actions = [0.0, 0.0, 0.0, 0.0]
        # # self.states = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        # #                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        # #                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        # self.states = []
        # self.rewards = [0.0, 0.0, 0.0, 0.0]

        self.actions = []
        self.states = []
        self.rewards = []
        self.old_state = np.array(np.random.uniform(low=-0.05, high=0.05, size=(6,)))

    def buildNetwork(self):
        """ Assemble shared layers
        """
        inp = Input((self.env_dim))
        x = Flatten()(inp)
        x = Dense(64, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        return Model(inp, x)

    def policy_action(self, s):
        """ Use the actor to predict the next action to take, using the policy
        """
        return np.random.choice(np.arange(self.act_dim), 1, p=self.actor.predict(s).ravel())[0]

    def discount(self, r):
        """ Compute the gamma-discounted rewards over an episode
        """
        discounted_r, cumul_r = np.zeros_like(r), 0
        for t in reversed(range(0, len(r))):
            cumul_r = r[t] + cumul_r * self.gamma
            discounted_r[t] = cumul_r
        return discounted_r

    def train_models(self, states, actions, rewards):
        """ Update actor and critic networks from experience
        """
        # Compute discounted rewards and Advantage (TD. Error)
        discounted_rewards = self.discount(rewards)
        state_values = self.critic.predict(np.array(states))
        advantages = discounted_rewards - np.reshape(state_values, len(state_values))
        # Networks optimization
        self.a_opt([states, actions, advantages])
        self.c_opt([states, discounted_rewards])

    def learn(self, new_state, action, reward, done):
        # Memorize (s, a, r) for training
        self.actions.append(to_categorical(action, self.act_dim))
        self.rewards.append(reward)
        self.states.append(new_state)
        # self.states = np.append(self.states, [new_state], axis=0)

        if done:
            # Train using discounted rewards ie. compute updates
            self.train_models(self.states, self.actions, self.rewards)
            self.actions = []
            self.states = []
            self.rewards = []



    def save_weights(self, path):
        path += '_LR_{}'.format(self.lr)
        self.actor.save(path)
        self.critic.save(path)

    def load_weights(self, path_actor, path_critic):
        self.critic.load_weights(path_critic)
        self.actor.load_weights(path_actor)
