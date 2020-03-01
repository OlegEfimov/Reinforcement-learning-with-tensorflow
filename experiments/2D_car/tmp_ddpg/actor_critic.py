#!/usr/bin/env python
"""
actor_critic.py

Actor and critic models.

author: Ben Cottier (git: bencottier)
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from mlp import MLP
import tensorflow as tf
from tensorflow.keras.layers import Dense, Concatenate, Flatten
from tensorflow.keras import Model, Input
from tensorflow.keras.initializers import RandomUniform
import numpy as np


class RLEstimator(tf.keras.Model):
    """
    Base class for gradient-based parametric estimators used in 
    reinforcement learning algorithms.
    """
    def __init__(self, lr=None, **kwargs):
        """
        Initialize an RLEstimator.

        Args:
            lr: float. Learning rate for this estimator's optimizer.
        """
        super(RLEstimator, self).__init__()
        if lr is not None:
            self.optimizer = tf.keras.optimizers.Adam(lr)

    @tf.function
    def polyak_update(self, other, p):
        """
        Updates the variables of this estimator as a "polyak" average
        of the current values and the values of the corresponding 
        variables in `other`, weighted by `p`.
        """
        for (ws, wo) in zip(self.trainable_variables, other.trainable_variables):
            ws.assign(p * ws + (1 - p) * wo)


class Actor(RLEstimator):
    """
    Actor component of an actor-critic model. Equivalent to a 
    policy estimate function, and can be called as such.
    """
    def __init__(self, arch=MLP, hidden_sizes=(400, 300), activation='relu', 
            action_space=None, input_shape=None, **kwargs):
        """
        Initialize an Actor estimator.

        Args:
            arch: keras.Model. A parametric model architecture. 
                Must implement the same interface as mlp.MLP.
            hidden_sizes: tuple of int. Number of hidden units for each 
                layer in the model architecture.
            activation: str. Name of the hidden activation function in 
                the model architecture.
            action_space: gym.Space. Action space of the environment.
            input_shape: int or tuple of int. Shape of the tensor input
                to the model.
        """
        super(Actor, self).__init__(**kwargs)
        self.act_range = np.array([1.0])
        self.action_space = action_space
        act_dim = self.action_space.shape[0]
        # self.act_limit = self.action_space.high[0]
        self.act_limit = 1.0
        # self.model = arch(list(hidden_sizes) + [act_dim], activation, 'tanh', input_shape)
        self.model = self.network(input_shape[1], act_dim)
        self.model.summary()

    def network(self, input_size, act_dim):
        """ Actor Network for Policy function Approximation, using a tanh
        activation for continuous control. We add parameter noise to encourage
        exploration, and balance it with Layer Normalization.
        """
        # inp = tf.keras.Input((self.env_dim))
        inp = tf.keras.Input((input_size))
        #
        x = tf.keras.layers.Dense(60, activation='relu')(inp)
        x = tf.keras.layers.GaussianNoise(1.0)(x)
        #
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(40, activation='relu')(x)
        x = tf.keras.layers.GaussianNoise(1.0)(x)
        x = tf.keras.layers.Dropout(0.5)(x)

        # out = tf.keras.layers.Dense(self.act_dim, activation='tanh', kernel_initializer=tf.keras.initializers.RandomUniform())(x)
        out = tf.keras.layers.Dense(act_dim, activation='tanh', kernel_initializer=tf.keras.initializers.RandomUniform())(x)
        out = tf.keras.layers.Lambda(lambda i: i * self.act_range)(out)

        return tf.keras.Model(inp, out)

    def save(self, path):
        # self.model.summary()
        # self.model.save_weights('model_actor.h5')
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        tflite_model = converter.convert()
        open("converted_model_tf1.tflite", "wb").write(tflite_model)

# Convert the model.
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()



    @tf.function
    def call(self, x):
        return self.act_limit * self.model(x)


class Critic(RLEstimator):
    """
    Critic component of an actor-critic model. Equivalent to a 
    Q estimate function, and can be called as such.
    """
    def __init__(self, arch=MLP, hidden_sizes=(400, 300), 
            activation='relu', input_shape=None, **kwargs):
        """
        Initialize a Critic estimator.

        Args:
            arch: keras.Model. A parametric model architecture. 
                Must implement the same interface as mlp.MLP.
            hidden_sizes: tuple of int. Number of hidden units for each 
                layer in the model architecture.
            activation: str. Name of the hidden activation function in 
                the model architecture.
            input_shape: int or tuple of int. Shape of the tensor input
                to the model.
        """
        super(Critic, self).__init__(**kwargs)
        # self.model = arch(list(hidden_sizes) + [1], activation, None, input_shape)
        self.model = self.network(input_shape[1])
        # self.model = self.network()
        self.model.summary()

    def network(self, input_size):
    # def network(self):

        """ Assemble Critic network to predict q-values
        """
        # state = Input((self.env_dim))
        # action = Input((self.act_dim,))
        state_action = Input((7,))
        state = Input((6,))
        action = Input((1,))
        x = Dense(80, activation='relu')(state_action)
        # x = Concatenate([Flatten()(x), action])
        x = Dense(80, activation='relu')(x)
        x = Dense(70, activation='relu')(x)
        x = Dense(60, activation='relu')(x)
        x = Dense(50, activation='relu')(x)
        out = Dense(1, activation='linear', kernel_initializer=RandomUniform())(x)
        # return Model([state, action], out)
        return Model([state_action], out)

    def save(self, path):
        print('critic save')
        # self.model.summary()
        # self.model.save_weights('model_critic.h5')
        # converter = tf.lite.TFLiteConverter.from_keras_model_file('model_actor.h5')
        # tflite_model = converter.convert()
        # open("converted_model_tf1.tflite", "wb").write(tflite_model)

    @tf.function
    def call(self, x, a):
        return tf.squeeze(self.model(tf.concat([x, a], axis=1)), axis=1)
