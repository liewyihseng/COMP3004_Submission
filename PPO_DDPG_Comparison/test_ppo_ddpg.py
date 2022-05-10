import matplotlib.pyplot as plt
import numpy as np

from PPO_DDPG_Comparison.utils import eval_models


PPO_rewards_eval = eval_models("../PPO_DDPG_Comparison/Training/Saved_Models/PPO_Model")
DDPG_rewards_eval = eval_models("../PPO_DDPG_Comparison/Training/Saved_Models/DDPG_Model", Model="DDPG")

x = np.linspace(1, len(PPO_rewards_eval), len(PPO_rewards_eval))

plt.plot(x, PPO_rewards_eval, "r", label="PPO")
plt.plot(x, DDPG_rewards_eval, "g", label="DDPG")
plt.xlabel("Number of Episodes")
plt.ylabel("Rewards")
plt.legend()
plt.title("Comparison of Rewards of Models Trained using PPO and DDPG")
plt.show()
