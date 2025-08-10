import gym
import numpy as np
from stable_baselines3 import PPO
from umr_rl_env import UMRSimulationEnv

# Register custom env if needed
env = UMRSimulationEnv()

# Train PPO agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=20000)

# Save model
model.save("umr_rl_agent")

# Evaluate
obs = env.reset()
for _ in range(10):
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    env.render()
