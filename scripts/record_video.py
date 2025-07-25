import imageio
import numpy as np
from stable_baselines3 import PPO
from blood_vessel_env import BloodVesselEnv


def record_agent_video(model_path, video_path, max_steps=500):
    env = BloodVesselEnv()
    model = PPO.load(model_path)

    obs, _ = env.reset()
    frames = []

    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)

        # Use env.render() to get the frame (OpenCV BGR), convert to RGB
        env.renderer.update_scene(env.data)
        frame = env.renderer.render()
        frames.append(frame)

        if done:
            break

    env.close()

    # Save as video, fps=30
    imageio.mimsave(video_path, frames, fps=30)
    print(f"Video saved to: {video_path}")


if __name__ == "__main__":
    record_agent_video("ppo_blood_vessel.zip", "blood_vessel_navigation.mp4")
