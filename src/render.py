"""
    Stage to generate a gif of the environment with the trained agent.
"""

from agents.ddpg import DDPG
from utils.anns import actor, critic
from utils.action_noise import OUActionNoise
from utils.buffer import Buffer
from utils.params import get_params
from utils.visualization import save_frames_as_gif
import gymnasium as gym
import numpy as np


params = get_params('gym_test')
ENVIRONMENT = str(params['environment'])
params = get_params()
EPISODES = params['episodes']

# %%

env = gym.make(ENVIRONMENT, render_mode='rgb_array')

num_states = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]


actor_model = actor(num_states, num_actions, 1e-3)
critic_model = critic(num_states, num_actions, 1e-3)

buffer = Buffer(num_states, num_actions, batch_size=32,
                buffer_capacity=int(1024))
noise = OUActionNoise(mean=np.zeros(num_actions),
                      sigma=float(0.2) * np.ones(num_actions))
ddpg = DDPG(env, buffer, actor_model, critic_model,
            act_noise=noise, tau=3e-3)

ddpg.load_actor_weights(f"models/actor_ddpg-{ENVIRONMENT}.hdf5")
ddpg.load_critic_weights(f"models/critic_ddpg-{ENVIRONMENT}.hdf5")

# %%

_, _, _, frames = ddpg.evaluate(1, visualize=True)

save_frames_as_gif(frames, path='evaluation', filename=f'{ENVIRONMENT}.gif')

# %%

ddpg.env = gym.make(ENVIRONMENT, render_mode='human')
a = ddpg.evaluate(EPISODES)
print(a)
