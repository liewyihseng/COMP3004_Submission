import matplotlib.pyplot as plt
import numpy as np
from PPO_DDPG_Comparison.utils import eval_models

PPO1Mil_rewards_eval = eval_models("../PPO_DDPG_Comparison/Training/Saved_Models/PPO_Model_200K")
PPO2mil_rewards_eval = eval_models("../PPO_DDPG_Comparison/Training/Saved_Models/PPO_Model_2Mil")

x = np.linspace(1, len(PPO1Mil_rewards_eval), len(PPO2mil_rewards_eval))

plt.plot(x, PPO1Mil_rewards_eval, "g", label="PPO (1000000 timesteps)")
plt.plot(x, PPO2mil_rewards_eval, "r", label="PPO (2000000 timesteps)")
plt.xlabel("Number of Episodes")
plt.ylabel("Rewards")
plt.legend()
plt.title("Comparison of Rewards of Models Trained Using PPO of Different Timesteps")
plt.show()
