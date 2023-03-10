"""
    Buffer memory implementation for experience replay.
"""
import numpy as np
import tensorflow as tf


class Buffer(object):
    """
        Buffer memory implementation for experience replay.

        Based on implementation from:
        https://github.com/keras-team/keras-io/blob/master/examples/rl/ddpg_pendulum.py

        Args:

        num_states (int): Dimension of the state space
        num_actions (int): Dimension of the input space
        buffer_capacity (int): Maximum number of samples to be stored. After
                               this number of samples, FIFO is implemented.
                               Default: 1e6.
        batch_size (int): Number of samples in each sampled batch.
                            Default: 64.

    """
    def __init__(self, num_states: int, num_actions: int,
                 buffer_capacity: int = int(1e6), batch_size: int = 64):

        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.num_states = num_states
        self.buffer_counter = 0

        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):

        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    def read(self):

        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size,
                                         replace=True)
        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(
            self.next_state_buffer[batch_indices])

        return state_batch, action_batch, reward_batch, next_state_batch
