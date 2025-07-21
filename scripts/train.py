from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from blood_vessel_env import BloodVesselEnv

# 建立環境
env = BloodVesselEnv()
check_env(env)  # 檢查格式是否正確

# 訓練 PPO 模型
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000)

# 儲存模型
model.save("ppo_blood_vessel")
