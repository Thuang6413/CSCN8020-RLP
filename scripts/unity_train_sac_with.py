from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from unity_blood_vessel_env import BloodVesselEnv
import numpy as np
import multiprocessing
import gymnasium as gym


def make_env(rank, seed=0):
    def _init():
        env = BloodVesselEnv(render_mode=None)  # No rendering in MuJoCo
        env.reset(seed=seed + rank)
        return env
    return _init


if __name__ == '__main__':
    # Create environment
    try:
        num_envs = multiprocessing.cpu_count()
        env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
        check_env(make_env(0)())
    except Exception as e:
        print(f"Error initializing environment: {e}")
        exit(1)

    # Initialize SAC model
    try:
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=1e-3,
            buffer_size=1000000,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            ent_coef="auto"
        )
    except Exception as e:
        print(f"Error initializing SAC model: {e}")
        env.close()
        exit(1)

    # Training loop without rendering
    total_timesteps = 1000000
    timestep = 0
    obs = env.reset()

    try:
        while timestep < total_timesteps:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, done, info = env.step(action)
            timestep += 1

            # Log progress for the first environment
            if timestep % 1000 == 0:
                print(
                    f"Timestep: {timestep}, Clot Distance: {info[0]['clot_dist']:.4f}, Reward: {reward[0]:.4f}")

            # Reset environments if done
            if any(done):
                obs = env.reset()

    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        # Save model and close environment
        model.save("sac_blood_vessel")
        env.close()
