import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from mujoco import mj_step
import cv2
from mujoco import Renderer
import os


class BloodVesselEnv(gym.Env):
    def __init__(self, space_size=1.2, vessel_radius=0.1, viscosity=15e-3, friction_coeff=1.0, render_mode="rgb_array"):
        super().__init__()
        self.render_mode = render_mode
        xml_path = "../assets/blood_vessel_merge.xml"
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XML file not found at {xml_path}")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # --- Get IDs ---
        entrance_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "vessel_entrance")
        exit_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "vessel_exit")
        target_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        self.mocap_id = self.model.body_mocapid[target_id]
        self.entrance_pos = self.model.body_pos[entrance_id]
        self.exit_pos = self.model.body_pos[exit_id]

        # --- Environment parameters ---
        self.space_size = space_size
        self.vessel_radius = vessel_radius
        self.friction_coeff = friction_coeff
        self.max_magnetic_field = 10.0
        self.max_steps = 1000

        # Initialize viscosity and flow_vel, which will be randomized in reset
        self.viscosity = viscosity
        self.flow_vel = 0

        if self.render_mode == "human" or self.render_mode == "rgb_array":
            self.camera_names = ["top_view", "side_view", "front_view"]
            self.renderers = {cam: Renderer(
                self.model, width=640, height=480) for cam in self.camera_names}
        else:
            self.renderers = None

        # --- Action space and observation space (unchanged) ---
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)

        self.target_pos = np.zeros(3)
        self.step_count = 0

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        # --- Stage 3 core modification: environment randomization ---
        # Randomize environment physical parameters at the beginning of each episode
        # 1. Randomize flow rate (within a reasonable range)
        flow_rate = np.random.uniform(low=1.5e-6, high=3.5e-6)
        # 2. Randomize viscosity
        self.viscosity = np.random.uniform(low=0.010, high=0.025)
        # 3. Recalculate flow velocity based on new flow rate
        self.flow_vel = flow_rate / (self.vessel_radius ** 2)
        # --- End of modification ---

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = self.entrance_pos.copy()
        self.data.qvel[:] = 0.0

        # Randomize target position (unchanged)
        self.target_pos = np.random.uniform(
            low=-self.vessel_radius, high=self.vessel_radius, size=3)
        self.target_pos[2] = np.random.uniform(
            low=self.entrance_pos[2], high=self.exit_pos[2])
        while np.sqrt(self.target_pos[0]**2 + self.target_pos[1]**2) > self.vessel_radius:
            self.target_pos[:2] = np.random.uniform(
                low=-self.vessel_radius, high=self.vessel_radius, size=2)

        self.data.mocap_pos[self.mocap_id] = self.target_pos
        self.step_count = 0

        return self._get_obs(), {}

    def _get_obs(self):
        vector_to_target = self.target_pos - self.data.qpos[:3]
        return np.concatenate([
            vector_to_target,
            self.data.qvel[:3],
            np.array([self.flow_vel, self.viscosity])
        ]).astype(np.float32)

    def step(self, action):
        # The logic of the step function is exactly the same as stage 2, since the task objective has not changed
        self.step_count += 1
        B_magnitude = (action[0] + 1) * self.max_magnetic_field / 2
        B_dir = action[1:4] / np.linalg.norm(
            action[1:4]) if np.linalg.norm(action[1:4]) > 0 else np.zeros(3)
        force = B_magnitude * B_dir * 1e-3
        self.data.ctrl[:3] = force
        flow_force = -self.flow_vel * self.viscosity * self.friction_coeff
        self.data.qfrc_applied[:3] = [flow_force, 0, 0]

        radial_distance = np.sqrt(self.data.qpos[0]**2 + self.data.qpos[1]**2)
        distance_outside_radial = max(0, radial_distance - self.vessel_radius)
        z_min, z_max = self.entrance_pos[2], self.exit_pos[2]
        distance_outside_z = 0
        if self.data.qpos[2] < z_min:
            distance_outside_z = z_min - self.data.qpos[2]
        elif self.data.qpos[2] > z_max:
            distance_outside_z = self.data.qpos[2] - z_max

        mj_step(self.model, self.data)
        obs = self._get_obs()
        current_dist = np.linalg.norm(self.target_pos - self.data.qpos[:3])

        goal_reward = 0
        # --- Precision training modification ---
        # Reduce the target radius, requiring the agent to reach more precisely
        terminated = bool(current_dist < 0.02)  # Reduced from 0.05 to 0.02
        # --- End of modification ---
        if terminated:
            goal_reward = 1000

        distance_reward = 10 * (1 - np.tanh(10 * current_dist))
        time_penalty = -0.05
        outside_penalty = 0
        if distance_outside_radial > 0 or distance_outside_z > 0:
            outside_penalty = -500
            terminated = True

        reward = goal_reward + distance_reward + time_penalty + outside_penalty

        if self.step_count >= self.max_steps:
            terminated = True

        truncated = False
        info = {"clot_dist": current_dist, "flow_vel": self.flow_vel,
                "outside_penalty": outside_penalty}
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode not in ["human", "rgb_array"] or self.renderers is None:
            return None

        images = []
        for cam in self.camera_names:
            self.renderers[cam].update_scene(self.data, camera=cam)
            img = self.renderers[cam].render()
            img_bgr = np.ascontiguousarray(img[..., ::-1])

            clot_dist = np.linalg.norm(self.data.qpos[:3] - self.target_pos)
            cv2.putText(img_bgr, f"Dist: {clot_dist:.4f} m ({cam})", (
                10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(img_bgr, f"Flow: {self.flow_vel*1000:.2f} mm/s",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            images.append(img_bgr)

        combined_img = np.hstack(images)
        if self.render_mode == "human":
            cv2.imshow("Blood Vessel Navigation", combined_img)
            cv2.waitKey(1)
        return combined_img

    def close(self):
        if self.render_mode == "human":
            cv2.destroyAllWindows()
