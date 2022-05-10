import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from PPO_DDPG_Comparison.extract_rgb import RObservation
from PPO_DDPG_Comparison.utils import environment_name, make_env

'''
In charge of running the PPO model trained using the original settings with 1mil timesteps
'''

env = make_env()
env = Monitor(env)
# Loads the previously saved model
model = PPO.load("../PPO_DDPG_Comparison/Training/Saved_Models/PPO_Model", env=env)
reward, std = evaluate_policy(model, env, n_eval_episodes=10, render=True)
env.close()
