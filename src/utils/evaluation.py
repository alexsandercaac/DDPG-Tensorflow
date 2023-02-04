"""
    Module with functions to evaluate the performance of the agent.
"""
import rich
from tqdm.rich import tqdm
import numpy as np


def evaluate(policy, env, episodes: int = 5, visualize: str = 'gif',
             old_api: bool = True) -> tuple:
    """
    Evaluate a RL agent
    :param policy: (function) function that, given a state, returns an action.
    :env: (gym.env) gym environment
    :episodes: (int) number of episodes to evaluate the agent
    :visualize: (str)one of 'gif', 'window' or 'none'
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
        if old_api:
            obs = env.reset()
            action = policy(obs)[0]
        else:
            obs, _ = env.reset()
            action = policy(obs)

        pbar.update(1)
        while not done:
            if visualize == 'gif':
                if old_api:
                    frames.append(env.render(mode='rgb_array'))
                else:
                    frames.append(env.render())
            elif visualize == 'window' and old_api:
                env.render(mode='human')
            if old_api:
                state, reward, done, _ = env.step(action)
                action = policy(state)[0]
            else:
                state, reward, terminated, truncated, _ = env.step(action)
                action = policy(state)
                done = terminated or truncated
            steps += 1
            episode_rewards.append(reward)
        episode_rewards_list.append(sum(episode_rewards))
        episode_len_list.append(steps)

    if visualize:
        env.close()
    mean_episode_reward = np.mean(episode_rewards_list)
    std_episode_reward = np.std(episode_rewards_list)
    mean_episode_len = np.mean(episode_len_list)
    if visualize:
        return (mean_episode_reward, std_episode_reward, mean_episode_len,
                frames)
    else:
        return mean_episode_reward, std_episode_reward, mean_episode_len
