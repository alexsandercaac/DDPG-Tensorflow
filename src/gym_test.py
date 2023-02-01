"""
    Test the DDPG agent on the basic gym environments
"""
# %%
from agents.ddpg import DDPG
from utils.anns import actor_bnorm, critic_bnorm, actor, critic
from utils.action_noise import OUActionNoise
from utils.buffer import Buffer
from utils.params import get_params
import gymnasium as gym
import numpy as np
import json

# %% Parameters

params = get_params()

CRITIC_LR = float(params["critic_lr"])
ACTOR_LR = float(params["actor_lr"])
ENVIRONMENT = str(params['environment'])
SAVE_MODELS = bool(params['save_models'])
BATCH_SIZE = int(float(params['batch_size']))
BUFFER_CAPACITY = int(float(params['buffer_capacity']))
SIGMA = float(params['sigma'])
TAU = float(params['tau'])
N_TRAINING_STEPS = int(float(params['n_training_steps']))
WARM_UP_STEPS = int(float(params['warm_up_steps']))
CLIP_GRADIENTS = bool(params['clip_gradients'])
LOG_FREQ = int(float(params['log_freq']))
EVAL_EPISODES = int(float(params['eval_episodes']))
BNORM = bool(params['bnorm'])
SAVE_BEST = bool(params['save_best'])

# %%
env = gym.make(ENVIRONMENT)

num_states = env.observation_space.shape[0]
print("Size of state space ->  {}".format(num_states))
num_actions = env.action_space.shape[0]
print("Size of action space ->  {}".format(num_actions))

upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]

print(f"Max, min value of action ->  {upper_bound}, {lower_bound}")

if BNORM:
    actor_model = actor_bnorm(num_states, num_actions, ACTOR_LR)
    critic_model = critic_bnorm(num_states, num_actions, CRITIC_LR)
else:
    actor_model = actor(num_states, num_actions, ACTOR_LR)
    critic_model = critic(num_states, num_actions, CRITIC_LR)

buffer = Buffer(num_states, num_actions, batch_size=BATCH_SIZE,
                buffer_capacity=int(BUFFER_CAPACITY))
noise = OUActionNoise(mean=np.zeros(num_actions),
                      sigma=float(SIGMA) * np.ones(num_actions))
ddpg = DDPG(env, buffer, actor_model, critic_model,
            act_noise=noise, tau=TAU)
# ddpg.load_actor_weights(f"agents/actor_ddpg-{problem}.hdf5")
# ddpg.load_critic_weights(f"agents/critic_ddpg-{problem}.hdf5")
# %%

hist = ddpg.fit(int(N_TRAINING_STEPS), warm_up=WARM_UP_STEPS,
                clip_grad=CLIP_GRADIENTS, log_freq=LOG_FREQ,
                eval_episodes=EVAL_EPISODES)

if SAVE_BEST:
    ddpg.target_actor.set_weights(ddpg.best_actor_weights)
    ddpg.target_critic.set_weights(ddpg.best_critic_weights)

ret = ddpg.evaluate(EVAL_EPISODES, visualize=False)
print("Evaluation:", ret)
metrics = {'mean_reward': ret[0], 'mean_episode_length': ret[1]}
with open(f"evaluation/metrics_ddpg-{ENVIRONMENT}.json", 'w') as f:
    json.dump(metrics, f)

ddpg.env = gym.make(ENVIRONMENT, render_mode="human")
ddpg.evaluate(2, visualize=True)

if SAVE_MODELS:
    ddpg.save_actor_weights(f"models/actor_ddpg-{ENVIRONMENT}.hdf5")
    ddpg.save_critic_weights(f"models/critic_ddpg-{ENVIRONMENT}.hdf5")
