import gym
from gym import spaces
import numpy as np
from my_simulation import VesselEnvironment, UMRProperties


class UMRSimulationEnv(gym.Env):
    def __init__(self):
        super(UMRSimulationEnv, self).__init__()

        # Instantiate the environment model
        self.vessel = VesselEnvironment()
        self.umr = UMRProperties()

        # Define action space: e.g., [UMR frequency (Hz)] in range [1, 15]
        self.action_space = spaces.Box(
            low=np.array([1.0]), high=np.array([15.0]), dtype=np.float32
        )

        # Observation space: e.g., Rcyl/Rves and other vascular states (can expand)
        self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)

        self.state = None
        self.reset()

    def step(self, action):
        umr_freq = float(action[0])
        self.umr.umr_freq = umr_freq
        self.umr.umr_speed_9_hz = 0.1 * umr_freq  # example effect

        # Extract resistance and other parameters
        r1 = self.vessel.rcylrves_renal_small_umr or 1.0
        r2 = self.vessel.rcylrves_aorta_large_umr or 1.0
        umr_speed = self.umr.umr_speed_9_hz

        # Compute your reward as before
        effect = 0.1 * umr_freq - r1
        reward = -abs(effect)

        # Normalize parameters (example: dividing by max expected values)
        norm_r1 = r1 / 10
        norm_r2 = r2 / 10
        norm_umr_speed = umr_speed / 2  # depends on max expected value

        # Create a state vector with multiple features
        self.state = np.array(
            [norm_r1, norm_r2, norm_umr_speed, umr_freq / 15, reward], dtype=np.float32
        )

        done = False
        return self.state, reward, done, {}

    def reset(self):
        self.vessel = VesselEnvironment()
        self.umr = UMRProperties()
        self.state = np.zeros(5, dtype=np.float32)
        return self.state

    def render(self, mode="human"):
        print(f"UMR Freq: {self.umr.umr_freq} | State: {self.state}")
