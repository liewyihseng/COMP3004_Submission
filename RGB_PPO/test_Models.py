import matplotlib.pyplot as plt
import numpy as np
from PPO_DDPG_Comparison.utils import eval_models

Green_rewards_eval = eval_models("../RGB_PPO/Training/Saved_Models/PPO_Green_Model", "Green")
Red_rewards_eval = eval_models("../RGB_PPO/Training/Saved_Models/PPO_Red_Model", "Red")
Blue_rewards_eval = eval_models("../RGB_PPO/Training/Saved_Models/PPO_Blue_Model", "Blue")

x = np.linspace(1, len(Green_rewards_eval), len(Green_rewards_eval))

plt.plot(x, Green_rewards_eval, "g", label="Green Channel")
plt.plot(x, Red_rewards_eval, "r", label="Red Channel")
plt.plot(x, Blue_rewards_eval, "b", label="Blue Channel")
plt.xlabel('Number of Episode')
plt.ylabel("Rewards")
plt.legend()
plt.title("Comparison of Rewards of Models Trained Using RGB Layers")
plt.show()


