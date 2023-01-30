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
    inputs_bnorm1 = tf.keras.layers.BatchNormalization()(inputs)
    regularizer = tf.keras.regularizers.l2(1e-2)
    dense1 = tf.keras.layers.Dense(256, activation="linear",
                                   kernel_regularizer=regularizer
                                   )(inputs_bnorm1)
    bnorm1 = tf.keras.layers.BatchNormalization()(dense1)
    act1 = tf.keras.layers.Activation('relu')(bnorm1)

    dense2 = tf.keras.layers.Dense(128, activation="linear",
                                   kernel_regularizer=regularizer)(act1)
    bnorm2 = tf.keras.layers.BatchNormalization()(dense2)
    act2 = tf.keras.layers.Activation('relu')(bnorm2)

    dense3 = tf.keras.layers.Dense(64, activation="linear",
                                   kernel_regularizer=regularizer)(act2)
    bnorm3 = tf.keras.layers.BatchNormalization()(dense3)
    act3 = tf.keras.layers.Activation('relu')(bnorm3)

    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
    outputs = tf.keras.layers.Dense(num_actions, activation="tanh",
                                    kernel_initializer=last_init)(act3)

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
    state_bnorm1 = tf.keras.layers.BatchNormalization()(state_input)
    regularizer = tf.keras.regularizers.l2(1e-2)
    state_dense1 = tf.keras.layers.Dense(256, activation="linear",
                                         kernel_regularizer=regularizer
                                         )(state_bnorm1)
    state_bnorm2 = tf.keras.layers.BatchNormalization()(state_dense1)
    state_act1 = tf.keras.layers.Activation('relu')(state_bnorm2)

    state_dense2 = tf.keras.layers.Dense(128, activation="linear",
                                         kernel_regularizer=regularizer
                                         )(state_act1)
    state_bnorm3 = tf.keras.layers.BatchNormalization()(state_dense2)
    state_act2 = tf.keras.layers.Activation('relu')(state_bnorm3)

    # Action as input
    action_input = tf.keras.layers.Input(shape=(num_actions))
    action_dense1 = tf.keras.layers.Dense(128, activation="linear",
                                          kernel_regularizer=regularizer
                                          )(action_input)
    action_bnorm1 = tf.keras.layers.BatchNormalization()(action_dense1)
    action_act1 = tf.keras.layers.Activation('relu')(action_bnorm1)

    action_dense2 = tf.keras.layers.Dense(128, activation="linear",
                                          kernel_regularizer=regularizer
                                          )(action_act1)
    action_bnorm2 = tf.keras.layers.BatchNormalization()(action_dense2)
    action_act2 = tf.keras.layers.Activation('relu')(action_bnorm2)

    # Final block
    concat = tf.keras.layers.Concatenate()([state_act2, action_act2])

    dense1 = tf.keras.layers.Dense(64, activation="relu",
                                   kernel_regularizer=regularizer)(concat)
    dropout1 = tf.keras.layers.Dropout(0.2)(dense1)

    dense2 = tf.keras.layers.Dense(64, activation="relu",
                                   kernel_regularizer=regularizer)(dropout1)
    dropout2 = tf.keras.layers.Dropout(0.2)(dense2)

    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
    outputs = tf.keras.layers.Dense(1, kernel_initializer=last_init)(dropout2)

    model = tf.keras.Model([state_input, action_input], outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr))

    return model
