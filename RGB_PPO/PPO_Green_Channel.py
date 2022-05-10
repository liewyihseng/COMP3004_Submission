import os

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

from PPO_DDPG_Comparison.extract_rgb import GObservation
from PPO_DDPG_Comparison.utils import environment_name


def make_env():
    env = gym.make(environment_name)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = GObservation(env)
    env = gym.wrappers.FrameStack(env, 4)
    return env


config = {
    "policy_type": "CnnPolicy",
    "total_timesteps": 1000000,
    "env_name": environment_name,
    "learning_rate": 0.00003
}

env = DummyVecEnv([make_env])
env = Monitor(env)

env = VecVideoRecorder(env,
                       f"videos/PPO",
                       record_video_trigger=lambda x: x % 2000 == 0,
                       video_length=200)

checkpoint_callback = CheckpointCallback(save_freq=2000, save_path=f"checkpoint_models/PPO")

# Training of PPO
model = PPO(config["policy_type"], env, verbose=1,
            learning_rate=config['learning_rate'],
            tensorboard_log=f"runs/Models")

model.learn(total_timesteps=config["total_timesteps"], callback=checkpoint_callback)

ppo_green_path = os.path.join("Training", "Saved_Models", "PPO_Green_Model")
model.save(ppo_green_path)
