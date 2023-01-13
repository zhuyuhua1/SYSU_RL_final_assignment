import numpy as np
import os
import tensorflow.compat.v1 as tf
from dopamine.colab import utils as colab_utils
import pickle
from absl import flags
LOG_PATH2 = 'tmp_mdqn_c/m_dqn_2023-01-13_16-32-32_000728'
LOG_PATH3 = 'tmp_dqn_c/dqn_2023-01-13_16-58-18_289709'
LOG_PATH = 'tmp_mddqn/m_dqn_2023-01-13_15-56-24_029226'
import seaborn as sns
import matplotlib.pyplot as plt
random_dqn_data = colab_utils.read_experiment(LOG_PATH,iteration_number=199,summary_keys=('train_episode_returns',), verbose=True)
random_dqn_data2 = colab_utils.read_experiment(LOG_PATH2,iteration_number=199,summary_keys=('train_episode_returns',), verbose=True)
random_dqn_data3 = colab_utils.read_experiment(LOG_PATH3,iteration_number=199,summary_keys=('train_episode_returns',), verbose=True)


print(random_dqn_data)
rewards = list(random_dqn_data['train_episode_returns'])[:75]
rewards2 =list(random_dqn_data2['train_episode_returns'])[:75]
rewards3 =list(random_dqn_data3['train_episode_returns'])[:75]
x1=np.arange(0,len(rewards),1)
x2=np.arange(0,len(rewards2),1)
x3=np.arange(0,len(rewards3),1)
plt.plot(x1, rewards, label="M-DDQN")
plt.plot(x2, rewards2, label="M-DQN")
plt.plot(x3, rewards3, label="DQN")
plt.legend()
plt.ylabel("rewards")
plt.xlabel("iterations")
plt.show()


