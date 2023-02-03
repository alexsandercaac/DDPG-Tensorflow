"""
    Test the DDPG agent on the basic gym environments
"""
# %%
from utils.params import get_params
import gym
import numpy as np
import json
from utils.action_noise import OUActionNoise
from stable_baselines3 import DDPG
from stable_baselines3.common.evaluation import evaluate_policy
import time
# %% Parameters

params = get_params()

LR = float(params["lr"])
ENVIRONMENT = str(params['environment'])
SAVE_MODELS = bool(params['save_models'])
BATCH_SIZE = int(float(params['batch_size']))
BUFFER_CAPACITY = int(float(params['buffer_capacity']))
SIGMA = float(params['sigma'])
TAU = float(params['tau'])
N_TRAINING_STEPS = int(float(params['n_training_steps']))
LOG_FREQ = int(float(params['log_freq']))
EVAL_EPISODES = int(float(params['eval_episodes']))
WARM_UP_STEPS = int(float(params['warm_up_steps']))
LEARN_FREQ = int(float(params['learn_freq']))
# %%
env = gym.make(ENVIRONMENT)

num_states = env.observation_space.shape[0]
print("Size of state space ->  {}".format(num_states))
num_actions = env.action_space.shape[0]
print("Size of action space ->  {}".format(num_actions))

upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]

print(f"Max, min value of action ->  {upper_bound}, {lower_bound}")

action_noise = OUActionNoise(
    mean=np.zeros(num_actions), sigma=SIGMA * np.ones(num_actions))

model = DDPG('MlpPolicy', env, verbose=1, action_noise=action_noise, tau=TAU,
             buffer_size=BUFFER_CAPACITY, batch_size=BATCH_SIZE,
             learning_rate=LR, train_freq=(1, 'step'))

start = time.time()
model.learn(total_timesteps=N_TRAINING_STEPS, log_interval=LOG_FREQ,
            progress_bar=True)
end = time.time()
time_taken = end - start
if SAVE_MODELS:
    model.save(f"models/ddpg-{ENVIRONMENT}-baseline.zip")
    print(f"Saved model to models/ddpg-{ENVIRONMENT}-baseline.zip")

eval = evaluate_policy(model, env, n_eval_episodes=EVAL_EPISODES, render=False)
print(f"Evaluated model on {ENVIRONMENT} environment")
print(eval)

metrics = {'mean_reward': eval[0], 'std_reward': eval[1],
           'time_taken_to_train_s': time_taken}

with open(f"evaluation/metrics_ddpg-{ENVIRONMENT}-baseline.json", 'w') as f:
    json.dump(metrics, f)

# %%
