"""
    Test the DDPG agent on the basic gym environments
"""
# %%
from agents.ddpg import DDPG
from utils.anns import actor_bnorm, critic_bnorm
from utils.action_noise import OUActionNoise
from utils.buffer import Buffer
import gymnasium as gym
import numpy as np

# %% Learning rate for actor-critic models
CRITIC_LR = 5e-4
ACTOR_LR = 2e-4
ENVIRONMENT = "Pendulum-v1"

env = gym.make(ENVIRONMENT)

num_states = env.observation_space.shape[0]
print("Size of state space ->  {}".format(num_states))
num_actions = env.action_space.shape[0]
print("Size of action space ->  {}".format(num_actions))

upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]

print(f"Max, min value of action ->  {upper_bound}, {lower_bound}")

actor_model = actor_bnorm(num_states, num_actions, ACTOR_LR)
critic_model = critic_bnorm(num_states, num_actions, CRITIC_LR)

buffer = Buffer(num_states, num_actions, batch_size=64,
                buffer_capacity=int(50e3))
noise = OUActionNoise(mean=np.zeros(num_actions),
                      sigma=float(0.2) * np.ones(num_actions))
ddpg = DDPG(env, buffer, actor_model, critic_model,
            act_noise=noise, tau=3e-3)
# ddpg.load_actor_weights(f"agents/actor_ddpg-{problem}.hdf5")
# ddpg.load_critic_weights(f"agents/critic_ddpg-{problem}.hdf5")
# %%

hist = ddpg.fit(int(5e4), warm_up=1e3, clip_grad=True, log_freq=50,
                eval_episodes=30)
ret = ddpg.evaluate(30, visualize=False)
print(ret)
ddpg.env = gym.make("Pendulum-v1", render_mode="human")
ddpg.evaluate(2, visualize=True)

ddpg.save_actor_weights(f"../models/actor_ddpg-{ENVIRONMENT}.hdf5")
ddpg.save_critic_weights(f"../models/critic_ddpg-{ENVIRONMENT}.hdf5")
