import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from PPO_DDPG_Comparison.utils import make_env, environment_name

'''
Handles the initialisation and training of the PPO algorithm
'''

# Initialisation of all the essential parameters
config = {
    "policy_type": "CnnPolicy",
    "total_timesteps": 1000000,
    "env_name": environment_name,
    "learning_rate": 0.00003
}

# Prepares the environment to have its performances monitored
env = DummyVecEnv([make_env])
env = Monitor(env)

# Keeps a record of the game play
env = VecVideoRecorder(env,
                       f"videos/PPO",
                       record_video_trigger=lambda x: x % 2000 == 0,
                       video_length=200)

# Constantly saving checkpoints during the training of the PPO algorithm
checkpoint_callback = CheckpointCallback(save_freq=2000, save_path=f"checkpoint_models/PPO")

# Initialisation of DDPG
model = PPO(config["policy_type"], env, verbose=1,
            learning_rate=config['learning_rate'],
            tensorboard_log=f"runs/Models")

# Training of DDPG Algorithm
model.learn(total_timesteps=config["total_timesteps"], callback=checkpoint_callback)

# Saves the resulting DDPG model into the "Training" file
ppo_path = os.path.join("Training", "Saved_Models", "PPO_Model_200K")
model.save(ppo_path)
