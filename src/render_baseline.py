"""
    Stage to generate a gif of the environment with the trained agent.
"""

from utils.params import get_params
import gym
from utils.evaluation import evaluate
from stable_baselines3 import DDPG
from utils.visualization import save_frames_as_gif


params = get_params('gym_test')
ENVIRONMENT = str(params['environment'])
params = get_params()
EPISODES = params['episodes']

# %%

env = gym.make(ENVIRONMENT)

model = DDPG.load(f"models/ddpg-{ENVIRONMENT}-baseline.zip")

# %%

_, _, _, frames = evaluate(lambda x: model.predict(x), env, episodes=1,
                           visualize='gif', old_api=True)

save_frames_as_gif(frames, path='evaluation',
                   filename=f'{ENVIRONMENT}-baseline.gif')

# %%

a = evaluate(lambda x: model.predict(x), env, episodes=EPISODES,
             visualize='window', old_api=True)
print(a)
