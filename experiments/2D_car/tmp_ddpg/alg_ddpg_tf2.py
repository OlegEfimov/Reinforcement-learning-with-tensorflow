from actor_critic import Actor, Critic
from replay_buffer import ReplayBuffer
import tensorflow as tf
import numpy as np
import os


class DDPG:
    """ Deep Deterministic Policy Gradient (DDPG) Helper Class
    """

    # def __init__(self, act_dim, env_dim, act_range, k, buffer_size = 2000, gamma = 0.99, lr = 0.0005, tau = 0.01):

    def __init__(act_dim, env_dim, act_range,
        ac_kwargs=dict(), seed=0, steps_per_epoch=5000, epochs=100, 
        replay_size=int(1e6), discount=0.99, polyak=0.995, pi_lr=1e-3, q_lr=1e-3, 
        batch_size=100, start_steps=10000, act_noise=0.1, max_ep_len=1000, 
        save_freq=1):

        self.time = 0
        # Set random seed for relevant modules
        tf.random.set_seed(seed)
        np.random.seed(seed)
        self.discount = discount
        self.act_noise = act_noise

        # Create environment
        # self.env, self.test_env = env_fn(), env_fn()
        self.obs_dim = env_dim
        self.act_dim = act_dim

        # Action limit for clipping
        # Assumes all dimensions have the same limit
        self.ac_kwargs = ac_kwargs
        # self.act_limit = self.env.action_space.high[0]
        self.act_limit = 1

        # Give actor-critic model access to action space
        self.ac_kwargs['action_space'] = act_range

        # Randomly initialise critic and actor networks
        self.critic = Critic(input_shape=(batch_size, self.obs_dim + self.act_dim), lr=q_lr, **self.ac_kwargs)
        self.actor = Actor(input_shape=(batch_size, self.obs_dim), lr=pi_lr, **self.ac_kwargs)

        # Initialise target networks with the same weights as main networks
        self.critic_target = Critic(input_shape=(batch_size, self.obs_dim + self.act_dim), **self.ac_kwargs)
        self.actor_target = Actor(input_shape=(batch_size, self.obs_dim), **self.ac_kwargs)
        self.critic_target.set_weights(self.critic.get_weights())
        self.actor_target.set_weights(self.actor.get_weights())

        # Initialise replay buffer for storing and getting batches of transitions
        self.replay_buffer = ReplayBuffer(self.obs_dim, self.act_dim, size=replay_size)

    def policy_action(self, o):
        """
        Computes an action from the policy (as a function of the 
        observation `o`) with added noise (scaled by `noise_scale`),
        clipped within the bounds of the action space.
        """
        a = self.actor(o.reshape(1, -1))
        a += self.act_noise * np.random.randn(self.act_dim)
        return np.clip(a, -self.act_limit, self.act_limit)

    @tf.function
    def train_step(self, batch):
        """
        Performs a gradient update on the actor and critic estimators
        from the given batch of transitions.

        Args:
            batch: dict. A batch of transitions. Must store valid 
                values for 'obs1', 'acts', 'obs2', 'rwds', and 'done'. 
                Obtained from ReplayBuffer.sample_batch().
        Returns:
            A tuple of the Q values, critic loss, and actor loss.
        """
        with tf.GradientTape(persistent=True) as tape:
            # Critic loss
            q = self.critic(batch['obs1'], batch['acts'])
            q_pi_targ = self.critic_target(batch['obs2'], self.actor_target(batch['obs2']))
            backup = tf.stop_gradient(batch['rwds'] + self.discount * (1 - batch['done']) * q_pi_targ)
            q_loss = tf.reduce_mean((q - backup)**2)
            # Actor loss
            pi = self.actor(batch['obs1'])
            q_pi = self.critic(batch['obs1'], pi)
            pi_loss = -tf.reduce_mean(q_pi)
        # Q learning update
        critic_gradients = tape.gradient(q_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))
        # Policy update
        actor_gradients = tape.gradient(pi_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
        return q, q_loss, pi_loss


    def learn(self, arg_old_state, arg_action, arg_reward, arg_new_state, arg_done):
        o =  arg_old_state
        o2 =  arg_new_state
        r = arg_reward
        d = arg_done
        ep_ret = arg_reward
        
        a = arg_action
        a = np.clip(a+self.noise.generate(self.time), -self.act_range, self.act_range)

        # Store transition in replay buffer
        replay_buffer.store(o, a, r, o2, d)

        batch = replay_buffer.sample_batch(batch_size)

        # Actor-critic update
        q, q_loss, pi_loss = train_step(batch)
        logger.store((max_logger_steps, batch_size), QVals=q.numpy())
        logger.store(max_logger_steps, LossQ=q_loss.numpy(), LossPi=pi_loss.numpy())

        # Target update
        critic_target.polyak_update(critic, polyak)
        actor_target.polyak_update(actor, polyak)
        self.time += 1
