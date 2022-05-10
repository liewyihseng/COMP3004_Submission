import gym
from numpy import mean
from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from PPO_DDPG_Comparison.extract_rgb import GObservation, BObservation, RObservation

environment_name = "CarRacing-v0"

# Preprocessing of Environment
def make_env():
    env = gym.make(environment_name)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, 4)
    return env


def eval_models(path, colour=None, Model="PPO"):
    env = gym.make(environment_name)
    env = Monitor(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if colour is None:
        colour = "Grey"
        env = gym.wrappers.GrayScaleObservation(env)
    elif colour is "Green":
        env = GObservation(env)
    elif colour is "Blue":
        env = BObservation(env)
    elif colour is "Red":
        env = RObservation(env)
    env = gym.wrappers.FrameStack(env, 4)
    if Model is "PPO":
        model = PPO.load(path, env=env)
    elif Model is "DDPG":
        model = DDPG.load(path, env=env)
    episode_rewards, episode_lengths = evaluate_policy(model,
                                                       env,
                                                       n_eval_episodes=100,
                                                       render=False,
                                                       return_episode_rewards=True)
    for i in range(len(episode_rewards)):
        print("Reward " + str(i) + ": " + str(episode_rewards[i]))
    print("The mean reward for " + colour + " Observation is " + str(mean(episode_rewards)))
    env.close()
    return episode_rewards


def eval_DDPG_models(path, colour=None, ):
    env = gym.make(environment_name)
    env = Monitor(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if colour is None:
        colour = "Grey"
        env = gym.wrappers.GrayScaleObservation(env)
    elif colour is "Green":
        env = GObservation(env)
    elif colour is "Blue":
        env = BObservation(env)
    elif colour is "Red":
        env = RObservation(env)
    env = gym.wrappers.FrameStack(env, 4)
    model = DDPG.load(path, env=env)
    episode_rewards, episode_lengths = evaluate_policy(model,
                                                       env,
                                                       n_eval_episodes=100,
                                                       render=False,
                                                       return_episode_rewards=True)
    for i in range(len(episode_rewards)):
        print("Reward " + str(i) + ": " + str(episode_rewards[i]))
    print("The mean reward for " + colour + " Observation is " + str(mean(episode_rewards)))
    env.close()
    return episode_rewards


