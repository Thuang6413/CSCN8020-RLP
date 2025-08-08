import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from mujoco import mj_step
import cv2
from mujoco import Renderer


class BloodVesselEnv(gym.Env):
    def __init__(self, space_size=0.05, viscosity=15e-3, friction_coeff=1.0, render_mode="rgb_array"):
        super().__init__()
        self.render_mode = render_mode
        # MuJoCo model with camera in <worldbody>
        self.model = mujoco.MjModel.from_xml_string("""
        <mujoco>
            <worldbody>
                <camera name="overview" pos="0 0 0.1" xyaxes="1 0 0 0 1 0" fovy="45"/>
                <body name="agent" pos="0 0 0">
                    <geom type="capsule" size="0.001875 0.0045" density="1000" rgba="0 0 1 1"/>
                    <freejoint name="agent_free"/>
                </body>
                <body name="target" mocap="true">
                    <geom type="sphere" size="0.002" rgba="1 0 0 1"/>
                </body>
                <body name="trajectory_root">
                    <!-- Placeholder for dynamic trajectory geoms -->
                </body>
            </worldbody>
            <actuator>
                <motor name="force_x" joint="agent_free"/>
                <motor name="force_y" joint="agent_free"/>
                <motor name="force_z" joint="agent_free"/>
            </actuator>
        </mujoco>
        """)
        self.data = mujoco.MjData(self.model)

        # Environment parameters
        self.space_size = space_size  # 50 mm = 0.05 m
        self.viscosity = viscosity  # 15 mPa·s = 0.015 Pa·s (fixed)
        self.friction_coeff = friction_coeff  # 0.4–1.6
        # 137.5 mL/min (fixed, middle of 15–260 mL/min)
        self.flow_rate = 2.29e-6
        self.flow_vel = self.flow_rate / (self.space_size ** 2)  # 0.916 mm/s
        self.max_speed = 19e-3  # 19 mm/s
        self.max_magnetic_field = 10.0  # mT
        self.max_steps = 1000

        # Initialize renderer
        if self.render_mode == "human" or self.render_mode == "rgb_array":
            self.renderer = Renderer(self.model, width=640, height=480)

        # Action space: [B_magnitude, B_x, B_y, B_z]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Observation space: [x, y, z, vx, vy, vz, flow_vel, viscosity, clot_dist, B_magnitude]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

        # Initialize flow, target, trajectory, and previous states
        self.target_pos = np.zeros(3)
        self.step_count = 0
        self.mocap_id = self.model.body(
            "target").mocapid  # Get mocap ID for target
        self.trajectory = []  # Store agent trajectory
        self.trajectory_root_id = self.model.body(
            "trajectory_root").id  # ID for trajectory root body
        # Max distance (diagonal ~86.6 mm)
        self.prev_clot_dist = np.sqrt(3 * (self.space_size ** 2))
        self.prev_pos = np.zeros(3)  # Previous position for 3D movement reward

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        mujoco.mj_resetData(self.model, self.data)
        # Fix agent and target on the z=0 plane
        # Place both agent and target at the same position (e.g., origin)
        pos = np.array([0.0, 0.0, 0.0])
        self.data.qpos[:3] = pos
        self.data.qvel[:] = 0.0
        self.target_pos = pos.copy()
        # Update mocap position
        self.data.mocap_pos[self.mocap_id] = self.target_pos
        self.step_count = 0
        self.trajectory = [self.data.qpos[:3].copy()]  # Initialize trajectory
        self.prev_clot_dist = np.sqrt(
            2 * (self.space_size ** 2))  # Reset to max distance (2D diagonal)
        # Initialize previous position
        self.prev_pos = self.data.qpos[:3].copy()
        return self._get_obs(), {}

    def _get_obs(self):
        clot_dist = np.linalg.norm(self.data.qpos[:3] - self.target_pos)
        B_magnitude = self.data.ctrl[0] * self.max_magnetic_field / \
            2 + self.max_magnetic_field / 2  # Map [-1, 1] to [0, 10] mT
        return np.concatenate([
            self.data.qpos[:3],  # Position
            self.data.qvel[:3],  # Velocity
            np.array([self.flow_vel, self.viscosity, clot_dist, B_magnitude])
        ]).astype(np.float32)

    def step(self, action):
        self.step_count += 1
        # Map action to physical values
        B_magnitude = (action[0] + 1) * \
            self.max_magnetic_field / 2  # [0, 10] mT
        B_dir = action[1:4] / np.linalg.norm(action[1:4]) if np.linalg.norm(
            action[1:4]) > 0 else np.zeros(3)  # Normalize direction
        force = B_magnitude * B_dir * 1e-3  # Scale to force
        self.data.ctrl[:3] = force  # Apply to agent

        # Apply blood flow force (viscous drag)
        flow_force = -self.flow_vel * self.viscosity * self.friction_coeff
        self.data.qfrc_applied[:3] = [flow_force, 0, 0]  # Flow in x-direction

        # Check boundary collision
        collision_force = 0
        for i in range(3):
            if abs(self.data.qpos[i]) > self.space_size / 2:
                collision_force += self.friction_coeff * \
                    abs(self.data.qpos[i] - np.sign(self.data.qpos[i])
                        * self.space_size / 2)
                self.data.qpos[i] = np.clip(
                    self.data.qpos[i], -self.space_size / 2, self.space_size / 2)

        # Step simulation
        mj_step(self.model, self.data)

        # Update target position for rendering
        self.data.mocap_pos[self.mocap_id] = self.target_pos

        # Update trajectory
        self.trajectory.append(self.data.qpos[:3].copy())
        if len(self.trajectory) > 100:  # Limit trajectory length
            self.trajectory.pop(0)

        # Get observation and reward
        obs = self._get_obs()
        clot_dist = np.linalg.norm(self.data.qpos[:3] - self.target_pos)
        current_pos = self.data.qpos[:3].copy()
        reward = 10 * (self.prev_clot_dist - clot_dist) / \
            0.001 if clot_dist < self.prev_clot_dist else 0  # Linear reward per 1 mm
        if np.all(np.abs(current_pos - self.prev_pos) > 0):  # Non-zero movement in x, y, z
            reward += 1
        if clot_dist < 0.00001:  # Within 0.01 mm
            reward += 100
        self.prev_clot_dist = clot_dist  # Update previous distance
        self.prev_pos = current_pos  # Update previous position

        # Termination conditions
        terminated = bool(clot_dist < 0.001 or self.step_count >= self.max_steps or np.linalg.norm(
            self.data.qpos[:3]) > self.space_size)
        truncated = False
        info = {"clot_dist": clot_dist, "flow_vel": self.flow_vel}

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode not in ["human", "rgb_array"]:
            return
        # Update scene
        self.renderer.update_scene(self.data, camera="overview")
        # Add trajectory geoms dynamically
        for i in range(1, len(self.trajectory)):
            # Calculate segment start and end points
            start = self.trajectory[i-1]
            end = self.trajectory[i]
            # Calculate midpoint and orientation
            midpoint = (start + end) / 2
            length = np.linalg.norm(end - start)
            if length == 0:
                continue
            direction = (end - start) / length
            # Create a temporary geom for the trajectory segment
            geom_id = self.renderer.scene.ngeom
            if geom_id < self.renderer.scene.maxgeom:
                self.renderer.scene.geoms[geom_id].type = mujoco.mjtGeom.mjGEOM_CYLINDER
                self.renderer.scene.geoms[geom_id].size = [
                    0.0005, 0.0005, length / 2]  # Thin cylinder
                self.renderer.scene.geoms[geom_id].rgba = [0, 1, 0, 1]  # Green
                self.renderer.scene.geoms[geom_id].pos = midpoint
                # Compute rotation matrix for orientation
                z_axis = np.array([0, 0, 1])
                if np.allclose(direction, z_axis) or np.allclose(direction, -z_axis):
                    self.renderer.scene.geoms[geom_id].mat = np.eye(3)
                else:
                    axis = np.cross(z_axis, direction)
                    angle = np.arccos(np.dot(direction, z_axis))
                    self.renderer.scene.geoms[geom_id].mat = self._rotation_matrix(
                        axis, angle)
                self.renderer.scene.ngeom += 1

        # Render image
        img = self.renderer.render()
        # Ensure image is uint8 and contiguous
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        img_bgr = np.ascontiguousarray(img[..., ::-1])  # Convert RGB to BGR
        # Add text overlay for info
        clot_dist = np.linalg.norm(self.data.qpos[:3] - self.target_pos)
        cv2.putText(img_bgr, f"Dist: {clot_dist:.4f} m", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img_bgr, f"Flow: {self.flow_vel*1000:.2f} mm/s",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if self.render_mode == "human":
            cv2.imshow("Blood Vessel Navigation", img_bgr)
            cv2.waitKey(1)
        return img_bgr if self.render_mode == "rgb_array" else None

    def _rotation_matrix(self, axis, angle):
        """Compute rotation matrix for given axis and angle."""
        axis = axis / np.linalg.norm(axis)
        c, s = np.cos(angle), np.sin(angle)
        t = 1 - c
        x, y, z = axis
        return np.array([
            [t*x*x + c, t*x*y - s*z, t*x*z + s*y],
            [t*x*y + s*z, t*y*y + c, t*y*z - s*x],
            [t*x*z - s*y, t*y*z + s*x, t*z*z + c]
        ])

    def close(self):
        if self.render_mode == "human":
            cv2.destroyAllWindows()
