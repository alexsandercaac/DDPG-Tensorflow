"""
    Implementation of Deep Deterministic Policy Gradient (DDPG) using
    Tensorflow 2.0 and Keras.
"""
import tensorflow as tf
import numpy as np
from tqdm.rich import tqdm
import rich


class DDPG(object):
    """
        Main class implementation for Deep Deterministic Policy Gradient (DDPG)
        by Lilicrap et al. 2015.

        Args:

        env (gym.env): gym (or gymnasium) environment.
        buffer (Buffer): Buffer class implementation for experience replay.
        actor (tf.keras.Model): keras ANN for the actor.
        critic (tf.keras.Model): keras ANN for the critic.
        act_noise (ActionNoise): action noise class for exploration.
        gamma (float): forgetting factor.
        tau (float): smoothing constant for target networks update.
    """

    def __init__(self, env, buffer, actor, critic, act_noise,
                 gamma: float = 0.99, tau: float = 1e-3) -> None:

        self.buffer = buffer
        self.gamma = gamma
        self.tau = tau
        self.env = env
        self.act_noise = act_noise
        self.act_upper_bound = env.action_space.high
        self.crt_grad_norm_list = [0]
        self.act_grad_norm_list = [0]
        self.actor_loss_list = [0]
        self.critic_loss_list = [0]
        self.best_avg_reward = -np.inf
        self.eval_episodes = None
        self.hist = {'mean_returns': [], 'std_returns': [], 'mean_lens': []}

        if not hasattr(actor, "optimizer"):
            raise ValueError('Actor must have an optimizer')
        self.actor_model = actor
        self.target_actor = tf.keras.models.clone_model(actor)
        self.best_actor_weights = actor.get_weights()

        if not hasattr(critic, "optimizer"):
            raise ValueError('Critic must have an optimizer')
        self.critic_model = critic
        self.target_critic = tf.keras.models.clone_model(critic)
        self.best_critic_weights = critic.get_weights()

    def policy(self, state, training=True):
        """
            Policy function to sample an action.

            Args:

            state (tuple): state observation used to determine next action.
            training (bool): whether or not the action should be noisy.

            Returns:
            action (np.array): action sampled from the policy.
        """
        # The network expects a batch of states, so we reshape the state
        # if it is not already a batch.
        if state.ndim < 2:
            state = state.reshape(1, self.buffer.num_states)
        # If training, add noise to the action.
        if training:
            sampled_action = tf.squeeze(self.actor_model(state))
            noise = self.act_noise()
            sampled_action = sampled_action.numpy() + noise
        else:
            sampled_action = tf.squeeze(self.target_actor(state))
        # Clip the action to the environment's action space.
        sampled_action = np.clip(sampled_action, -1, 1)
        sampled_action = tf.multiply(sampled_action, self.act_upper_bound)

        # If the action is a scalar, we reshape it.
        if not sampled_action.shape:
            action = [sampled_action]
        else:
            action = sampled_action

        return action

    @tf.function
    def update_target(self, target_weights, weights, tau):
        """
            Update the target networks using the soft update rule.

            Args:

            target_weights (list): list of target network weights.
            weights (list): list of network weights.
            tau (float): smoothing constant for target networks update.
        """
        for a, b in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

    @tf.function
    def sgd_on_batch(self, state_batch, action_batch, reward_batch,
                     next_state_batch, clip_grad, grad_norm):
        """
            Perform a single step of gradient descent on a batch of data.

            Args:

            state_batch (np.array): batch of states.
            action_batch (np.array): batch of actions.
            reward_batch (np.array): batch of rewards.
            next_state_batch (np.array): batch of next states.
            clip_grad (bool): whether or not to clip the gradient.
            grad_norm (float): gradient norm to clip to.
        """
        # * Update the critic network.
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch, training=True)
            target_actions = tf.multiply(target_actions, self.act_upper_bound)
            # Bellman equation using target networks.
            y = reward_batch + self.gamma * self.target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = self.critic_model(
                [state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))
        # The critic loss is calculated using the target networks and Bellman
        # equation.
        critic_grad = tape.gradient(
            critic_loss, self.critic_model.trainable_variables)

        # Clip the gradient if necessary. We also calculate the gradient norm
        # for logging purposes.
        if clip_grad:
            critic_grad, crt_norm = tf.clip_by_global_norm(critic_grad,
                                                           grad_norm)
        else:
            _, crt_norm = tf.clip_by_global_norm(critic_grad, 1)
        self.critic_model.optimizer.apply_gradients(
            zip(critic_grad, self.critic_model.trainable_variables)
        )
        # * Update the actor network.
        with tf.GradientTape() as tape:
            actions = self.actor_model(state_batch, training=True)
            actions = tf.multiply(actions, self.act_upper_bound)
            critic_value = self.critic_model(
                [state_batch, actions], training=True)
            # The actor loss is the negative of the critic value.
            # We want to maximize the critic value, which is equivalent to
            # minimizing the negative of the critic value.
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(
            actor_loss, self.actor_model.trainable_variables)
        if clip_grad:
            actor_grad, act_norm = tf.clip_by_global_norm(actor_grad,
                                                          grad_norm)
        else:
            _, act_norm = tf.clip_by_global_norm(actor_grad, 1)
        self.actor_model.optimizer.apply_gradients(
            zip(actor_grad, self.actor_model.trainable_variables)
        )
        return crt_norm, act_norm, critic_loss, actor_loss

    def fit(self, steps, max_steps_per_ep=np.Inf, log_freq=25,
            warm_up=50, verbose=1, clip_grad=True, learn_freq=1,
            eval_episodes=10, performance_th=np.Inf, grad_norm=5,
            checkpoints=False, checkpoint_path='agents/',
            keep_best: bool = True):
        """
            Train the agent.

            Args:

            steps (int): number of steps to train the agent for.
            max_steps_per_ep (int): maximum number of steps per episode.
            log_freq (int): frequency of logging. At this frequency, the
                target networks are used to evaluate the agent's performance
                for eval_episodes episodes.
            warm_up (int): number of steps to warm up the agent before
                training.
            verbose (int): verbosity level.
            clip_grad (bool): whether or not to clip the gradient.
            learn_freq (int): frequency of learning in number of steps. The
                agent learns every learn_freq steps.
            eval_episodes (int): number of episodes to evaluate the agent
                for.
            performance_th (float): threshold for the performance of the
                agent. If the agent's performance is above this threshold,
                the training is stopped.
            grad_norm (float): gradient norm to clip to.
            checkpoints (bool): whether or not to save checkpoints.
            checkpoint_path (str): path to save checkpoints.
            keep_best (bool): whether or not to keep the best performing
                model. If true, the best performing model is restored.
        """
        # To store reward history of each episode
        self.crt_grad_norm_list = [0] + [np.nan for _ in range(log_freq - 1)]
        self.act_grad_norm_list = [0] + [np.nan for _ in range(log_freq - 1)]
        self.actor_loss_list = [0] + [np.nan for _ in range(log_freq - 1)]
        self.critic_loss_list = [0] + [np.nan for _ in range(log_freq - 1)]
        self.eval_episodes = eval_episodes

        steps_taken = 0
        episode = 0
        pbar = tqdm(total=steps)
        while steps_taken < steps:

            prev_state, _ = self.env.reset()
            done = False
            ep_steps = 0
            while not done:

                action = self.policy(prev_state)

                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                steps_taken += 1
                ep_steps += 1
                pbar.update(1)
                if steps_taken >= steps:
                    break
                self.buffer.record((prev_state, action, reward, state))
                if (steps_taken > warm_up) and (steps_taken % learn_freq == 0):
                    self.learn(clip_grad, grad_norm)

                prev_state = state
                if ep_steps > max_steps_per_ep:
                    done = True
                    break
            episode += 1
            self.act_noise.reset()
            if (episode % log_freq == 0) and (episode != 0):
                if verbose > 0:
                    print(f"\nEpisode: {episode}")
                score = self.log_optimization_info(verbose)
                if score >= self.best_avg_reward:
                    self.best_avg_reward = score
                    self.best_actor_weights = self.target_actor.get_weights()
                    self.best_critic_weights = self.target_critic.get_weights()

                if checkpoints:
                    self.save_actor_weights(checkpoint_path +
                                            "checkpoint_actor.hdf5")
                    self.save_critic_weights(checkpoint_path +
                                             "checkpoint_critic.hdf5")
                if score > performance_th:
                    if verbose > 0:
                        print("\nPerformance goal reached!! :)")
                    break
        if keep_best:
            self.target_actor.set_weights(self.best_actor_weights)
            self.target_critic.set_weights(self.best_critic_weights)
        pbar.close()
        return self.hist

    def learn(self, clip_grad=True, grad_norm=5) -> None:
        state_batch, action_batch, reward_batch, next_state_batch \
            = self.buffer.read()
        crt_grad_norm, act_grad_norm, crt_loss, act_loss = \
            self.sgd_on_batch(state_batch, action_batch,
                              reward_batch, next_state_batch,
                              clip_grad, grad_norm)
        self.manage_optimization_lists(crt_grad_norm,
                                       act_grad_norm, crt_loss,
                                       act_loss)
        self.update_target(self.target_actor.variables,
                           self.actor_model.variables, self.tau)
        self.update_target(self.target_critic.variables,
                           self.critic_model.variables, self.tau)

    def evaluate(self, episodes=5, visualize=False):
        """
        Evaluate a RL agent
        :param model: (BaseRLModel object) the RL Agent
        :param num_episodes: (int) number of episodes to evaluate it
        :return: (float) Mean reward for the last num_episodes
        """
        # This function will only work for a single Environment

        episode_rewards_list = []
        episode_len_list = []
        if visualize:
            frames = []
        try:
            pbar = tqdm(total=episodes)
        except rich.errors.LiveError:
            pbar = tqdm(total=episodes, disable=True)
        print("Evaluating policy...")
        for _ in range(episodes):
            episode_rewards = []
            steps = 0
            done = False
            obs, _ = self.env.reset()
            action = self.policy(obs.reshape(1, self.buffer.num_states),
                                 training=False)
            pbar.update(1)
            while not done:
                if visualize:
                    frames.append(self.env.render())
                state, reward, terminated, truncated, _ = self.env.step(action)
                action = self.policy(state)
                done = terminated or truncated
                steps += 1
                episode_rewards.append(reward)
            episode_rewards_list.append(sum(episode_rewards))
            episode_len_list.append(steps)

        if visualize:
            self.env.close()
        mean_episode_reward = np.mean(episode_rewards_list)
        std_episode_reward = np.std(episode_rewards_list)
        mean_episode_len = np.mean(episode_len_list)
        if visualize:
            return (mean_episode_reward, std_episode_reward, mean_episode_len,
                    frames)
        else:
            return mean_episode_reward, std_episode_reward, mean_episode_len

    def manage_optimization_lists(self, crt_grad_norm, act_grad_norm, crt_loss,
                                  act_loss):

        self.crt_grad_norm_list.append(crt_grad_norm)
        self.crt_grad_norm_list.pop(0)
        self.act_grad_norm_list.append(act_grad_norm)
        self.act_grad_norm_list.pop(0)
        self.critic_loss_list.append(crt_loss)
        self.critic_loss_list.pop(0)
        self.actor_loss_list.append(act_loss)
        self.actor_loss_list.pop(0)

    def log_optimization_info(self, verbose):
        mean_return, std_return, mean_len = self.evaluate(
            self.eval_episodes)
        mean_crt_grad = np.nanmean(self.crt_grad_norm_list)
        mean_act_grad = np.nanmean(self.act_grad_norm_list)
        mean_crt_loss = np.nanmean(self.critic_loss_list)
        mean_act_loss = np.nanmean(self.actor_loss_list)
        self.hist['mean_returns'].append(mean_return)
        self.hist['std_returns'].append(std_return)
        self.hist['mean_lens'].append(mean_len)
        if verbose > 0:
            tqdm.write("\n-----------------------------------\n" +
                       f"\nMean return: {mean_return:.2f}" +
                       f"\nStd return: {std_return:.2}" +
                       f"\nMean length: {mean_len:.2f}" +
                       "\nCritic gradient norm: " +
                       f"{mean_crt_grad:.2e}" +
                       "\nActor gradient norm: " +
                       f"{mean_act_grad:.2e}" +
                       "\nCritic loss: " +
                       f"{mean_crt_loss:.2f}" +
                       "\nActor loss: " +
                       f"{mean_act_loss:.2f}" +
                       "\n-----------------------------------\n",
                       end="")
        return mean_return

    def load_actor_weights(self, path):
        self.target_actor.load_weights(path)
        self.actor_model.load_weights(path)

    def load_critic_weights(self, path):
        self.target_critic.load_weights(path)
        self.critic_model.load_weights(path)

    def save_actor_weights(self, path):
        self.target_actor.save_weights(path)

    def save_critic_weights(self, path):
        self.target_critic.save_weights(path)
