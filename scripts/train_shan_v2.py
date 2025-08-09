# train_blood_vessel.py
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from mujoco import mj_step, Renderer

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv  # windows-safe


class BloodVesselEnv(gym.Env):
    """
    Minimal, fixed environment:
      - Free-floating capsule-like body with a freejoint (no motors).
      - We apply forces via data.xfrc_applied in world frame.
      - Uniform flow along +X with Stokes drag F = -6πμR (v - u_flow).
      - Target is randomly placed away from the agent on z=0 plane.
      - Shaped reward for progress, heading, upstream effort, wall avoidance, energy, smoothness.
    """
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 60}

    def __init__(
        self,
        space_size=0.05,          # 5 cm box
        viscosity=15e-3,          # Pa·s
        friction_coeff=0.2,       # extra damping
        flow_rate=2.29e-6,        # m^3/s (~137.5 mL/min)
        render_mode=None,
        seed: int | None = None,
    ):
        super().__init__()
        self.render_mode = render_mode

        # Build a tiny MuJoCo model: free body + mocap target
        xml = """
<mujoco>
  <option timestep="0.002" gravity="0 0 0" integrator="RK4"/>
  <worldbody>
    <camera name="overview" pos="0 0 0.12" xyaxes="1 0 0 0 1 0" fovy="50"/>
    <body name="agent" pos="0 0 0">
      <!-- capsule size=(radius, half-length) -->
      <geom type="capsule" size="0.001875 0.0045" density="1000" rgba="0 0 1 1"/>
      <freejoint name="agent_free"/>
    </body>
    <body name="target" mocap="true">
      <geom type="sphere" size="0.002" rgba="1 0 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        # Renderer (create only if asked to render)
        self.renderer = None
        if self.render_mode in ("rgb_array", "human"):
            self.renderer = Renderer(self.model, width=640, height=480)

        # Physics / env params
        self.space_size = float(space_size)
        self.viscosity = float(viscosity)
        self.friction_coeff = float(friction_coeff)
        self.flow_rate = float(flow_rate)
        self.flow_vel = self.flow_rate / (self.space_size ** 2)  # uniform m/s along +X
        self.R_robot = 0.001875                                 # capsule radius (m)
        self.propulsion_gain = 0.03                              # N per unit "mag"
        self.max_steps = 5000

        # IDs
        self.agent_bid = self.model.body("agent").id
        self.target_mocap_id = self.model.body("target").mocapid

        # Spaces
        # action: [throttle in -1..1, dir_x, dir_y, dir_z in -1..1]
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        # obs: pos(3) vel(3) flow_vel(1) viscosity(1) dist_to_target(1) last_mag(1) = 10
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(10,), dtype=np.float32)

        # State trackers
        self.target_pos = np.zeros(3, dtype=np.float64)
        self.prev_dist = np.inf
        self._last_mag = 0.0
        self.step_count = 0
        self._prev_action = np.zeros(4, dtype=np.float64)

        # Seeding
        if seed is not None:
            np.random.seed(seed)

    # ----------------- Gym API -----------------

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        mujoco.mj_resetData(self.model, self.data)

        # Agent starts near center on z=0
        agent_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.data.qpos[:3] = agent_pos
        self.data.qvel[:] = 0.0

        # Target random on z=0, at least 1 cm away
        while True:
            target = np.random.uniform(-self.space_size/2, self.space_size/2, size=3)
            target[2] = 0.0
            if np.linalg.norm(target - agent_pos) > 0.01:
                break
        self.target_pos = target
        self.data.mocap_pos[self.target_mocap_id] = self.target_pos

        self.prev_dist = np.linalg.norm(agent_pos - target)
        self._last_mag = 0.0
        self.step_count = 0
        self._prev_action[:] = 0.0

        return self._get_obs(), {}

    def _get_obs(self):
        pos = self.data.qpos[:3].copy()
        vel = self.data.qvel[:3].copy()
        dist = np.linalg.norm(pos - self.target_pos)
        return np.array(
            [
                pos[0], pos[1], pos[2],
                vel[0], vel[1], vel[2],
                self.flow_vel,
                self.viscosity,
                dist,
                self._last_mag,
            ],
            dtype=np.float32,
        )

    def step(self, action):
        self.step_count += 1
        a = np.asarray(action, dtype=np.float32)

        # Parse action -> propulsion force
        mag = float((a[0] + 1.0) * 0.5)       # 0..1
        d = a[1:4].astype(np.float64)
        d /= (np.linalg.norm(d) + 1e-9)       # unit dir
        F_prop = self.propulsion_gain * mag * d
        self._last_mag = mag

        # Stokes drag (relative to uniform flow in +X)
        u_flow = np.array([self.flow_vel, 0.0, 0.0], dtype=np.float64)
        v = self.data.qvel[:3].copy()
        stokes = 6.0 * math.pi * self.viscosity * self.R_robot
        F_drag = -stokes * (v - u_flow)

        # Extra linear damping
        F_loss = -self.friction_coeff * v

        # Apply net external force in world frame
        self.data.xfrc_applied[self.agent_bid, :3] = F_prop + F_drag + F_loss
        self.data.xfrc_applied[self.agent_bid, 3:] = 0.0

        # Integrate
        mj_step(self.model, self.data)

        # Soft bounds: terminate if we fly out of the 5cm cube
        out_of_bounds = np.any(np.abs(self.data.qpos[:3]) > (self.space_size/2))

        # --- Reward shaping ---
        pos = self.data.qpos[:3].copy()
        vel = self.data.qvel[:3].copy()
        dist = np.linalg.norm(pos - self.target_pos) + 1e-9

        # 1) Progress (potential-based)
        progress = (self.prev_dist - dist)
        r_progress = 4.0 * progress  # stronger than before so it dominates early learning

        # 2) Heading alignment (points the swimmer at the target)
        to_goal = (self.target_pos - pos) / dist
        v_along_goal = float(np.dot(vel, to_goal))   # m/s toward goal
        r_heading = 0.5 * v_along_goal               # small, dense signal every step

        # 3) Against-flow credit when needed (helps overcome +X flow)
        goal_is_upstream = (self.target_pos[0] < pos[0] - 1e-6)
        r_upstream = 0.3 * (-vel[0]) if goal_is_upstream else 0.0

        # 4) Centerline / wall avoidance (soft penalty near the box edges)
        half = self.space_size / 2.0
        margin = 0.005  # 5 mm safety margin
        def edge_penalty(p):
            # 0 in center, rises steeply within 'margin' of the wall
            d = half - abs(p)
            return 0.0 if d > margin else (1.0 - d / margin)
        r_wall = -0.5 * (edge_penalty(pos[0]) + edge_penalty(pos[1]) + edge_penalty(pos[2]))

        # 5) Control/energy regularization (prefer low actuation)
        r_energy = -0.01 * (mag ** 2)

        # 6) Action smoothness (discourage jitter)
        act_change = np.linalg.norm(action - self._prev_action)
        r_smooth = -0.02 * act_change
        self._prev_action = action.copy()

        # Base alive bonus so it doesn’t spiral to zero when stuck
        r_alive = 0.001

        # Sum
        reward = r_progress + r_heading + r_upstream + r_wall + r_energy + r_smooth + r_alive

        # Success bonus
        success = dist < 0.006  # within 6 mm
        if success:
            reward += 40.0

        self.prev_dist = dist

        terminated = bool(success or out_of_bounds)
        truncated = bool(self.step_count >= self.max_steps)
        info = {
            "dist": float(dist),
            "r_progress": float(r_progress),
            "r_heading": float(r_heading),
            "r_upstream": float(r_upstream),
            "r_wall": float(r_wall),
            "r_energy": float(r_energy),
            "r_smooth": float(r_smooth),
            "success": bool(success),
        }

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode not in ("rgb_array", "human"):
            return None
        self.renderer.update_scene(self.data, camera="overview")
        img = self.renderer.render()
        if self.render_mode == "human":
            try:
                import cv2
                cv2.imshow("BloodVessel", img[..., ::-1])
                cv2.waitKey(1)
            except Exception:
                pass
        return img

    def close(self):
        # Clean up any windows if human rendering was used
        try:
            import cv2
            cv2.destroyAllWindows()
        except Exception:
            pass


# ----------------- Training script -----------------

def main():
    # Windows-safe: single-process vector env
    env = make_vec_env(
        BloodVesselEnv,
        n_envs=1,
        vec_env_cls=DummyVecEnv,
        env_kwargs=dict(render_mode=None),
    )

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=500_000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        ent_coef=0.05,           # <-- fixed as float, keeps exploration up early
        # tensorboard_log="./tb", # optional: turn on TB logging
    )

    model.learn(total_timesteps=300_000, progress_bar=True)
    model.save("sac_blood_vessel")
    env.close()

    # Optional: visualize a short rollout after training
    vis = BloodVesselEnv(render_mode="rgb_array")
    obs, _ = vis.reset()
    try:
        import cv2
        for _ in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = vis.step(action)
            frame = vis.render()
            if frame is not None:
                cv2.imshow("Blood Vessel (Eval)", frame[..., ::-1])
                cv2.waitKey(1)
            if term or trunc:
                obs, _ = vis.reset()
        cv2.destroyAllWindows()
    except Exception:
        pass
    vis.close()


if __name__ == "__main__":
    main()
