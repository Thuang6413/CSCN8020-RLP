from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from blood_vessel_env import BloodVesselEnv

# Create environment
env = BloodVesselEnv()
check_env(env)  # Check if the environment follows the correct format

# Train PPO model
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000)

# Save model
model.save("ppo_blood_vessel")
