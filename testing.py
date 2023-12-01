import gym
import torch
env = gym.make("CartPole-v1")
observation,dummy = env.reset()
print("Initial observations:", observation)
result= env.step(0)
print(result)
observation, reward, done, info, dummy = env.step(0)
print("New observations after choosing action 0:", observation)
print("Reward for this step:", reward)
print("Is this round done?", done)

observation = env.reset()
cumulative_reward = 0
done = False
while not done:
    observation, reward, done, info,dummy = env.step(0)
    cumulative_reward += reward
print("Cumulative reward for this round:", cumulative_reward)