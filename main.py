import gym
import numpy as np
from wandbHelper import *
from REINFORCE import REINFORCE

wandb_init(project='monte-carlo',
           entity='skiing-rl',
           learning_rate=0.001,
           epochs=100,
           batch_size=128)

np.random.seed(0)
env = gym.make('Skiing-v4', render_mode="human")
env.reset()
s_dim = env.observation_space.shape
a_dim = env.action_space.n

net= CNN(s_dim, a_dim)
agent =REINFORCE(net)

cnt = 0
reward_sum = 0
while True:
    _, reward, done, _ = env.step(env.action_space.sample())
    cnt += 1

    if done:
        break
    # wandb_log({
    #     'loss': 10,
    #     'reward': reward
    # })
    reward_sum += reward
    print(reward, reward_sum, done)
print(cnt)
env.close()
