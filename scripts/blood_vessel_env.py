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
        # Check if XML file exists
        xml_path = "../assets/blood_vessel_merge.xml"
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XML file not found at {xml_path}")
        # MuJoCo model with camera in <worldbody>
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Print available cameras for debugging
        for i in range(self.model.ncam):
            cam_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            print(f"Available camera: {cam_name}")

        # Get body positions using mj_name2id
        entrance_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "vessel_entrance")
        exit_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "vessel_exit")
        if entrance_id == -1:
            raise ValueError("Body 'vessel_entrance' not found in the model")
        if exit_id == -1:
            raise ValueError("Body 'vessel_exit' not found in the model")
        self.entrance_pos = self.model.body_pos[entrance_id]
        self.exit_pos = self.model.body_pos[exit_id]

        # Get mocap ID for target
        target_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        if target_id == -1:
            raise ValueError("Body 'target' not found in the model")
        if self.model.body_mocapid[target_id] == -1:
            raise ValueError(
                "Body 'target' is not a mocap body (missing mocap='true' in XML)")
        # Get the actual mocap ID
        self.mocap_id = self.model.body_mocapid[target_id]
        # Check mocap_pos size
        if self.mocap_id >= self.data.mocap_pos.shape[0]:
            raise ValueError(
                f"Mocap ID {self.mocap_id} exceeds mocap_pos size {self.data.mocap_pos.shape[0]}")

        # Environment parameters
        self.space_size = space_size  # 1.2 m to match vessel length
        self.vessel_radius = vessel_radius  # 0.1 m to match vessel radius
        self.viscosity = viscosity  # 15 mPa·s = 0.015 Pa·s (fixed)
        self.friction_coeff = friction_coeff  # 0.4–1.6
        # 137.5 mL/min (fixed, middle of 15–260 mL/min)
        self.flow_rate = 2.29e-6
        # Adjust flow velocity based on vessel radius
        self.flow_vel = self.flow_rate / (self.vessel_radius ** 2)
        self.max_speed = 19e-3  # 19 mm/s
        self.max_magnetic_field = 10.0  # mT
        self.max_steps = 5000
        self.wall_collision_penalty = -3  # Penalty for hitting or staying at vessel walls
        self.boundary_threshold = 0.05  # 5% of vessel radius for boundary detection

        # Initialize renderers for three cameras
        self.camera_names = ["top_view", "side_view", "front_view"]
        if self.render_mode == "human" or self.render_mode == "rgb_array":
            self.renderers = {
                cam: Renderer(self.model, width=640, height=480) for cam in self.camera_names
            }
        else:
            self.renderers = None

        # Action space: [B_magnitude, B_x, B_y, B_z]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Observation space: [x, y, z, vx, vy, vz, flow_vel, viscosity, clot_dist, B_magnitude]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

        # Initialize flow, target, trajectory, and previous states
        self.target_pos = np.zeros(3)
        self.step_count = 0
        self.trajectory = []  # Store agent trajectory
        self.trajectory_root_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "trajectory_root")
        if self.trajectory_root_id == -1:
            raise ValueError("Body 'trajectory_root' not found in the model")
        # Max distance (diagonal of the cubic space)
        self.prev_clot_dist = np.sqrt(3 * (self.space_size ** 2))
        self.prev_pos = np.zeros(3)  # Previous position for 3D movement reward

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            self.np_random = np.random.RandomState(seed)
        else:
            self.np_random = np.random.RandomState()

        mujoco.mj_resetData(self.model, self.data)
        # Place agent at the vessel entrance
        self.data.qpos[:3] = self.entrance_pos.copy()
        self.data.qvel[:] = 0.0

        # Set target to a random position within the vessel
        # X and Y within vessel radius
        self.target_pos = self.np_random.uniform(
            low=-self.vessel_radius,
            high=self.vessel_radius,
            size=3
        )
        # Z within vessel length (between entrance and exit)
        self.target_pos[2] = self.np_random.uniform(
            low=self.entrance_pos[2],
            high=self.exit_pos[2]
        )
        # Ensure target is within cylindrical bounds
        while np.sqrt(self.target_pos[0]**2 + self.target_pos[1]**2) > self.vessel_radius:
            self.target_pos[:2] = self.np_random.uniform(
                low=-self.vessel_radius,
                high=self.vessel_radius,
                size=2
            )
        # Update mocap position for target
        self.data.mocap_pos[self.mocap_id] = self.target_pos

        self.step_count = 0
        self.trajectory = [self.data.qpos[:3].copy()]  # Initialize trajectory
        # Reset to max distance (3D diagonal)
        self.prev_clot_dist = np.sqrt(3 * (self.space_size ** 2))
        # Initialize previous position
        self.prev_pos = self.data.qpos[:3].copy()
        result = (self._get_obs(), {})
        # print(f"Reset return: {len(result)} values - {result}")
        return result

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

        # Check boundary collision and apply penalty for staying at or near boundary
        collision_penalty = 0
        # Check X and Y against vessel radius (cylindrical boundary)
        radial_distance = np.sqrt(self.data.qpos[0]**2 + self.data.qpos[1]**2)
        if radial_distance >= self.vessel_radius * (1 - self.boundary_threshold):
            collision_penalty += self.wall_collision_penalty
            # Project position back to cylindrical boundary
            if radial_distance > self.vessel_radius:
                scale = self.vessel_radius / \
                    (radial_distance + 1e-6)  # Avoid division by zero
                self.data.qpos[0] *= scale
                self.data.qpos[1] *= scale
        # Check Z against vessel length
        z_min = self.entrance_pos[2] + self.boundary_threshold * \
            (self.exit_pos[2] - self.entrance_pos[2])
        z_max = self.exit_pos[2] - self.boundary_threshold * \
            (self.exit_pos[2] - self.entrance_pos[2])
        if self.data.qpos[2] <= z_min or self.data.qpos[2] >= z_max:
            collision_penalty += self.wall_collision_penalty
            self.data.qpos[2] = np.clip(
                self.data.qpos[2], self.entrance_pos[2], self.exit_pos[2])

        # Log collision details for debugging
        # print(f"Step {self.step_count}: radial_distance={radial_distance:.6f}, z={self.data.qpos[2]:.6f}, "
        #       f"z_min={z_min:.6f}, z_max={z_max:.6f}, collision_penalty={collision_penalty}")

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
        # Exponential distance reward
        distance_reward = 1500 * \
            (1 - np.exp(-10 * (self.prev_clot_dist - clot_dist))
             ) if clot_dist < self.prev_clot_dist else 0
        if clot_dist < 0.00001:  # Within 0.01 mm
            goal_reward = 1000
        else:
            goal_reward = 0
        reward = distance_reward + goal_reward + \
            collision_penalty  # Removed time_penalty for simplicity
        self.prev_clot_dist = clot_dist  # Update previous distance
        self.prev_pos = current_pos  # Update previous position

        # Termination conditions
        terminated = bool(clot_dist < 0.001 or self.step_count >= self.max_steps or
                          np.linalg.norm(self.data.qpos[:3]) > self.space_size)
        truncated = False
        info = {"clot_dist": clot_dist, "flow_vel": self.flow_vel,
                "collision_penalty": collision_penalty}

        # Log reward components for debugging
        # print(f"Step {self.step_count}: distance_reward={distance_reward:.2f}, goal_reward={goal_reward:.2f}, "
        #       f"collision_penalty={collision_penalty:.2f}, total_reward={reward:.2f}")

        # Debug return values
        result = (obs, reward, terminated, truncated, info)
        # print(f"Step return: {len(result)} values - {result}")
        return result

    def render(self):
        if self.render_mode not in ["human", "rgb_array"]:
            return None
        if not hasattr(self, 'renderers') or self.renderers is None:
            return None  # Skip rendering if renderers are not initialized

        # Render from three camera perspectives
        images = []
        for cam in self.camera_names:
            try:
                self.renderers[cam].update_scene(self.data, camera=cam)
                img = self.renderers[cam].render()
                if img.dtype != np.uint8:
                    img = img.astype(np.uint8)
                img_bgr = np.ascontiguousarray(
                    img[..., ::-1])  # Convert RGB to BGR
                # Add text overlay for info
                clot_dist = np.linalg.norm(
                    self.data.qpos[:3] - self.target_pos)
                cv2.putText(img_bgr, f"Dist: {clot_dist:.4f} m ({cam})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(img_bgr, f"Flow: {self.flow_vel*1000:.2f} mm/s",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                images.append(img_bgr)
            except ValueError as e:
                print(f"Rendering error for camera {cam}: {e}")
                return None

        # Combine images horizontally
        combined_img = np.hstack(images)

        # Add trajectory geoms dynamically for all renderers
        for cam in self.camera_names:
            renderer = self.renderers[cam]
            for i in range(1, len(self.trajectory)):
                start = self.trajectory[i-1]
                end = self.trajectory[i]
                midpoint = (start + end) / 2
                length = np.linalg.norm(end - start)
                if length == 0:
                    continue
                direction = (end - start) / length
                geom_id = renderer.scene.ngeom
                if geom_id < renderer.scene.maxgeom:
                    renderer.scene.geoms[geom_id].type = mujoco.mjtGeom.mjGEOM_CYLINDER
                    renderer.scene.geoms[geom_id].size = [
                        0.005, 0.005, length / 2]  # Thin cylinder
                    renderer.scene.geoms[geom_id].rgba = [0, 1, 0, 1]  # Green
                    renderer.scene.geoms[geom_id].pos = midpoint
                    z_axis = np.array([0, 0, 1])
                    if np.allclose(direction, z_axis) or np.allclose(direction, -z_axis):
                        renderer.scene.geoms[geom_id].mat = np.eye(3)
                    else:
                        axis = np.cross(z_axis, direction)
                        angle = np.arccos(np.dot(direction, z_axis))
                        renderer.scene.geoms[geom_id].mat = self._rotation_matrix(
                            axis, angle)
                    renderer.scene.ngeom += 1

        if self.render_mode == "human":
            cv2.imshow("Blood Vessel Navigation", combined_img)
            cv2.waitKey(1)
        return combined_img

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
