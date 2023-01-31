"""
    Actor and Critic artificial neural networks (ANNs) for the DDPG algorithm.
"""
import tensorflow as tf


def actor(num_states: int, num_actions: int, lr: float = 1e-3
          ) -> tf.keras.Model:
    """
    Actor ANN for the DDPG algorithm.

    Args:
        num_states (int): Number of states.
        num_actions (int): Number of actions.
        lr (float, optional): Learning rate for the optimizer.
            Defaults to 1e-3.

    Returns:
        tf.keras.Model: Actor ANN.
    """

    inputs = tf.keras.layers.Input(shape=(num_states))

    dense1 = tf.keras.layers.Dense(256, activation="linear")(inputs)
    act1 = tf.keras.layers.Activation('relu')(dense1)

    dense2 = tf.keras.layers.Dense(256, activation="linear")(act1)
    act2 = tf.keras.layers.Activation('relu')(dense2)

    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
    outputs = tf.keras.layers.Dense(num_actions, activation="tanh",
                                    kernel_initializer=last_init)(act2)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr))

    return model


def critic(num_states: int, num_actions: int, lr: float = 1e-3
           ) -> tf.keras.Model:
    """
    Critic ANN for the DDPG algorithm.

    Args:
        num_states (int): Number of states.
        num_actions (int): Number of actions.
        lr (float, optional): Learning rate for the optimizer.
            Defaults to 1e-3.

    Returns:
        tf.keras.Model: Critic ANN.
    """

    # State as input
    state_input = tf.keras.layers.Input(shape=(num_states))
    regularizer = tf.keras.regularizers.l2(1e-2)
    state_dense1 = tf.keras.layers.Dense(16, activation="linear",
                                         kernel_regularizer=regularizer
                                         )(state_input)
    state_act1 = tf.keras.layers.Activation('relu')(state_dense1)
    state_dense2 = tf.keras.layers.Dense(32, activation="linear",
                                         kernel_regularizer=regularizer
                                         )(state_act1)
    state_act2 = tf.keras.layers.Activation('relu')(state_dense2)

    # Action as input
    action_input = tf.keras.layers.Input(shape=(num_actions))
    action_dense1 = tf.keras.layers.Dense(32, activation="linear",
                                          kernel_regularizer=regularizer
                                          )(action_input)
    action_act1 = tf.keras.layers.Activation('relu')(action_dense1)

    concat = tf.keras.layers.Concatenate()([state_act2, action_act1])

    dense1 = tf.keras.layers.Dense(256, activation="relu",
                                   kernel_regularizer=regularizer)(concat)
    dense2 = tf.keras.layers.Dense(256, activation="relu",
                                   kernel_regularizer=regularizer)(dense1)
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    outputs = tf.keras.layers.Dense(1, kernel_initializer=last_init)(dense2)

    model = tf.keras.Model([state_input, action_input], outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr))

    return model
