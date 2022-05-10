import os

from stable_baselines3 import DDPG
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.callbacks import CheckpointCallback
from PPO_DDPG_Comparison.utils import make_env, environment_name

'''
Handles the initialisation and training of the DDPG algorithm
'''

# Initialisation of all the essential parameters
config = {
    "policy_type": "CnnPolicy",
    "total_timesteps": 1000000,
    "env_name": environment_name,
    "actor_lr": 0.00003
}

# Prepares the environment to have its performances monitored
env = DummyVecEnv([make_env])
env = Monitor(env)

# Keeps a record of the game play
env = VecVideoRecorder(env,
                       f"videos/DDPG",
                       record_video_trigger=lambda x: x % 2000 == 0,
                       video_length=200)

# Constantly saving checkpoints during the training of the DDPG algorithm
checkpoint_callback = CheckpointCallback(save_freq=2000, save_path=f"checkpoint_models/DDPG")

# Initialisation of DDPG
model = DDPG(config["policy_type"], env,
             verbose=1,
             learning_rate= config["actor_lr"],
             tensorboard_log=f"runs/Models", buffer_size=30000)

# Training of DDPG Algorithm
model.learn(total_timesteps=config["total_timesteps"],
            callback=checkpoint_callback)

# Saves the resulting DDPG model into the "Training" file
ddpg_path = os.path.join("Training", "Saved_Models", "DDPG_Model")
model.save(ddpg_path)
