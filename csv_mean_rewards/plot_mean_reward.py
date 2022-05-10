import matplotlib.pyplot as plt
import pandas as pd

'''Plotting of Mean Reward Extracted from Tensorboard'''


'''Uncomment to plot the graph of DDPG and PPO Algorithm'''
# DDPG_df = pd.read_csv("../csv_mean_rewards/DDPG1Mil.csv")
# PPO_df = pd.read_csv("../csv_mean_rewards/PPO1Mil.csv")
#
# DDPG_df = DDPG_df.iloc[:, 1:3]
# PPO_df = PPO_df.iloc[:, 1:3]
#
# fig, ax = plt.subplots()
# ax.ticklabel_format(style='plain')
# plt.plot(DDPG_df["Step"], DDPG_df["Value"], "g", label="DDPG")
# plt.plot(PPO_df["Step"], PPO_df["Value"], "r", label="PPO")
# plt.xlabel("Timesteps")
# plt.ylabel("Mean Rewards")
# plt.legend()
# plt.title("Comparison of Mean Rewards of PPO and DDPG")
# plt.show()


'''Uncomment to plot the graphs of Mean Rewards obtained by each RGB channel'''
# green_df = pd.read_csv("../csv_mean_rewards/PPO_Green.csv")
# red_df = pd.read_csv("../csv_mean_rewards/PPO_Red.csv")
# blue_df = pd.read_csv("../csv_mean_rewards/PPO_Blue.csv")
#
# green_df = green_df.iloc[:, 1:3]
# red_df = red_df.iloc[:, 1:3]
# blue_df = blue_df.iloc[:, 1:3]
#
# fig, ax = plt.subplots()
# ax.ticklabel_format(style='plain')
# plt.plot(green_df["Step"], green_df["Value"], "-g", label="Green")
# plt.plot(red_df["Step"], red_df["Value"], "-r", label="Red")
# plt.plot(blue_df["Step"], blue_df["Value"], "-b", label="Blue")
# plt.xlabel("Timesteps")
# plt.ylabel("Mean Rewards")
# plt.legend()
# plt.title("Comparison of Mean Rewards of PPO Trained using Individual RGB Channel")
# plt.show()
