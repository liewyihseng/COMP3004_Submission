import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from RGB_PPO.extract_rgb import RObservation
from PPO_DDPG_Comparison.utils import environment_name

'''
In charge of running the best performing PPO model trained using the red channel
'''

'''
Initialisation of the environment

Output parameter:
    env: representing the environment where consisting only pixel values from the red colour channel
'''
def make_env():
    env = gym.make(environment_name)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = RObservation(env)
    env = gym.wrappers.FrameStack(env, 4)
    return env


env = make_env()
env = Monitor(env)
# Loads the previously saved model
model = PPO.load("../RGB_PPO/Training/Saved_Models/PPO_Red_Model", env=env)
reward, std = evaluate_policy(model, env, n_eval_episodes=10, render=True)
env.close()
