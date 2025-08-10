# visualize_policy.py
from stable_baselines3 import PPO
from blood_vessel_env import BloodVesselEnv
import time

env = BloodVesselEnv()
model = PPO.load("ppo_blood_vessel")  # Load the trained model

obs, _ = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, done, _, _ = env.step(action)
    env.render()  # Display animation
    if done:
        break
env.close()
