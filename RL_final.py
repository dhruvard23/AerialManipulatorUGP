"""
RL wrapper for UGP Dynamic Pick-and-Place

Purpose:
- Uses Reinforcement Learning to learn the pick/place timing parameters:
    center_frac, window_frac, hold_frac
- Keeps your existing cubic spline, energy proxy, IK, PID Lua controller unchanged.
- RL replaces optimize_arm_event_schedule() for event scheduling.

How it works:
1. Imports functions from your existing uap_final.py.
2. Builds optimized drone splines using your existing optimizer.
3. Trains an RL policy to choose:
       pick_center, pick_window, pick_hold,
       place_center, place_window, place_hold
4. Evaluates reward using your existing arm-energy and grasp-distance logic.
5. Runs final CoppeliaSim mission using the trained RL schedule.

Files needed:
- uap_final.py  -> your current full working script
- this file     -> uap_rl_schedule.py

Run training:
    python uap_rl_schedule.py --mode train

Run final mission using trained RL model:
    python uap_rl_schedule.py --mode run
"""

import argparse
import math
import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# Import your existing project code
import uap_final as base


RL_MODEL_PATH = "uap_rl_schedule_model"
RL_LOG_FILE = "uap_rl_training_log.csv"

np.random.seed(7)


# ============================================================
# Utility functions
# ============================================================

def normalize_value(x, lo, hi):
    return 2.0 * (x - lo) / max(hi - lo, 1e-9) - 1.0


def denormalize_action(a, lo, hi):
    a = float(np.clip(a, -1.0, 1.0))
    return lo + (a + 1.0) * 0.5 * (hi - lo)


def compute_event_guess(waypoints, object_world):
    """
    Estimate the normalized spline location where drone is closest in XY to object.
    """
    n = 180
    _, xyz = base.build_cubic_spline(waypoints, n)
    obj = np.asarray(object_world, dtype=float)
    xy_dists = np.linalg.norm(xyz[:, :2] - obj[:2], axis=1)
    event_guess = int(np.argmin(xy_dists)) / max(n - 1, 1)
    return float(event_guess)


def build_arm_trajectory_from_schedule(
    waypoints,
    object_world,
    q_start,
    q_end,
    speed,
    schedule,
    n=160,
):
    """
    Same logic as your optimize_arm_event_schedule(), but evaluates a schedule
    chosen by RL instead of optimizing with scipy.
    """
    _, xyz = base.build_cubic_spline(waypoints, n)

    path_len = float(np.sum(np.linalg.norm(np.diff(xyz, axis=0), axis=1)))
    tf = max(path_len / max(speed, 1e-3), 1.0)

    q_start = np.asarray(q_start, dtype=float)
    q_end = np.asarray(q_end, dtype=float)
    obj = np.asarray(object_world, dtype=float)

    center_frac = float(np.clip(schedule["center_frac"], 0.05, 0.95))
    window_frac = float(np.clip(schedule["window_frac"], 0.08, 0.50))
    hold_frac = float(np.clip(schedule["hold_frac"], 0.00, 0.12))

    half_window = window_frac / 2.0
    half_hold = hold_frac / 2.0

    t = np.linspace(0.0, 1.0, n)
    q_traj = np.zeros((n, 4), dtype=float)
    tip_dists = np.zeros(n, dtype=float)

    for k, tk in enumerate(t):
        q_nom = q_start + (q_end - q_start) * base.quintic(tk)

        local = obj - xyz[k]
        q_ik = base.ik_arm(local)

        d_event = abs(tk - center_frac)

        if d_event <= half_hold:
            blend = 1.0
        elif d_event <= half_window:
            blend_raw = 1.0 - (d_event - half_hold) / max(half_window - half_hold, 1e-6)
            blend = base.quintic(blend_raw)
        else:
            blend = 0.0

        q = (1.0 - blend) * q_nom + blend * q_ik
        q_traj[k] = q

        tip_world = xyz[k] + base.forward_tip_local(q)
        tip_dists[k] = float(np.linalg.norm(tip_world - obj))

    e_arm = base.estimate_arm_energy_torque(q_traj, tf)
    min_dist = float(np.min(tip_dists))

    qd = np.gradient(q_traj, tf / max(n - 1, 1), axis=0)
    max_q_speed = float(np.max(np.abs(qd)))

    motion_mag = float(
        np.trapz(
            np.sum((q_traj - q_start) ** 2, axis=1),
            dx=tf / max(n - 1, 1),
        )
    )

    return {
        "q_traj": q_traj,
        "tip_dists": tip_dists,
        "arm_energy": float(e_arm),
        "min_dist": float(min_dist),
        "max_q_speed": float(max_q_speed),
        "motion_mag": float(motion_mag),
        "tf": float(tf),
    }


def evaluate_schedule_cost(
    waypoints,
    object_world,
    q_start,
    q_end,
    speed,
    schedule,
    event_guess,
):
    """
    Cost used as RL negative reward.

    RL tries to minimize:
    - gripper-object distance
    - arm torque-energy
    - excessive arm movement
    - event timing too far from closest drone-object point
    - unnecessarily large event window
    - excessive joint speed
    """

    out = build_arm_trajectory_from_schedule(
        waypoints=waypoints,
        object_world=object_world,
        q_start=q_start,
        q_end=q_end,
        speed=speed,
        schedule=schedule,
    )

    min_dist = out["min_dist"]
    e_arm = out["arm_energy"]
    motion_mag = out["motion_mag"]
    max_q_speed = out["max_q_speed"]

    center_frac = schedule["center_frac"]
    window_frac = schedule["window_frac"]
    hold_frac = schedule["hold_frac"]

    grasp_cost = 2500.0 * min_dist**2
    energy_cost = 1.0 * e_arm
    motion_cost = 0.03 * motion_mag
    timing_cost = 15.0 * (center_frac - event_guess) ** 2
    window_cost = 0.15 * window_frac
    hold_cost = 0.05 * hold_frac
    speed_cost = 0.02 * max_q_speed**2

    total_cost = (
        grasp_cost
        + energy_cost
        + motion_cost
        + timing_cost
        + window_cost
        + hold_cost
        + speed_cost
    )

    return float(total_cost), out


# ============================================================
# RL Environment
# ============================================================

class UAPScheduleEnv(gym.Env):
    """
    One-step continuous-control RL environment.

    The agent outputs six continuous values:

        pick_center, pick_window, pick_hold,
        place_center, place_window, place_hold

    The environment evaluates these against the same physics-inspired
    arm-energy and grasp-distance metrics used in your original project.

    This is intentionally a high-level RL layer.
    It does not replace PID or spline optimization.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        home,
        pick,
        place,
        wps1,
        wps2,
        wps3,
        e1,
        e2,
        e3,
        randomize=False,
    ):
        super().__init__()

        self.base_home = np.asarray(home, dtype=float)
        self.base_pick = np.asarray(pick, dtype=float)
        self.base_place = np.asarray(place, dtype=float)

        self.wps1 = wps1
        self.wps2 = wps2
        self.wps3 = wps3

        self.e1 = float(e1)
        self.e2 = float(e2)
        self.e3 = float(e3)

        self.randomize = bool(randomize)

        # Action values are normalized in [-1, 1].
        # They are converted to actual schedule parameters below.
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32,
        )

        # Observation:
        # home xyz, pick xyz, place xyz,
        # pick_event_guess, place_event_guess,
        # drone energy proxies e1,e2,e3 normalized roughly.
        self.observation_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(14,),
            dtype=np.float32,
        )

        self.episode_id = 0
        self.logs = []

        self.current_home = self.base_home.copy()
        self.current_pick = self.base_pick.copy()
        self.current_place = self.base_place.copy()

        self.pick_guess = compute_event_guess(self.wps1, self.current_pick)
        self.place_guess = compute_event_guess(self.wps2, self.current_place)

    def _get_obs(self):
        obs = np.array(
            [
                self.current_home[0],
                self.current_home[1],
                self.current_home[2],
                self.current_pick[0],
                self.current_pick[1],
                self.current_pick[2],
                self.current_place[0],
                self.current_place[1],
                self.current_place[2],
                self.pick_guess,
                self.place_guess,
                self.e1 / 1000.0,
                self.e2 / 1000.0,
                self.e3 / 1000.0,
            ],
            dtype=np.float32,
        )
        return obs

    def _action_to_schedules(self, action):
        action = np.asarray(action, dtype=float)

        pick_schedule = {
            "center_frac": denormalize_action(action[0], 0.05, 0.95),
            "window_frac": denormalize_action(action[1], 0.08, 0.50),
            "hold_frac": denormalize_action(action[2], 0.00, 0.12),
        }

        place_schedule = {
            "center_frac": denormalize_action(action[3], 0.05, 0.95),
            "window_frac": denormalize_action(action[4], 0.08, 0.50),
            "hold_frac": denormalize_action(action[5], 0.00, 0.12),
        }

        return pick_schedule, place_schedule

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.episode_id += 1

        self.current_home = self.base_home.copy()
        self.current_pick = self.base_pick.copy()
        self.current_place = self.base_place.copy()

        # Optional small randomization so the policy does not overfit to one exact geometry.
        if self.randomize:
            rng = self.np_random
            self.current_pick[:2] += rng.uniform(-0.04, 0.04, size=2)
            self.current_place[:2] += rng.uniform(-0.04, 0.04, size=2)

        self.pick_guess = compute_event_guess(self.wps1, self.current_pick)
        self.place_guess = compute_event_guess(self.wps2, self.current_place)

        return self._get_obs(), {}

    def step(self, action):
        pick_schedule, place_schedule = self._action_to_schedules(action)

        pick_cost, pick_out = evaluate_schedule_cost(
            waypoints=self.wps1,
            object_world=self.current_pick,
            q_start=base.JOINT_FOLDED,
            q_end=base.JOINT_CARRY,
            speed=base.DESIRED_SPEED_EMPTY,
            schedule=pick_schedule,
            event_guess=self.pick_guess,
        )

        place_cost, place_out = evaluate_schedule_cost(
            waypoints=self.wps2,
            object_world=self.current_place,
            q_start=base.JOINT_CARRY,
            q_end=base.JOINT_CARRY,
            speed=base.DESIRED_SPEED_CARRY,
            schedule=place_schedule,
            event_guess=self.place_guess,
        )

        total_cost = pick_cost + place_cost

        # Success bonus if the gripper gets very close to object/drop location.
        pick_success = pick_out["min_dist"] < 0.08
        place_success = place_out["min_dist"] < 0.08

        success_bonus = 0.0
        if pick_success:
            success_bonus += 50.0
        if place_success:
            success_bonus += 50.0

        # Reward is negative cost.
        reward = -total_cost + success_bonus

        terminated = True
        truncated = False

        info = {
            "pick_center": pick_schedule["center_frac"],
            "pick_window": pick_schedule["window_frac"],
            "pick_hold": pick_schedule["hold_frac"],
            "place_center": place_schedule["center_frac"],
            "place_window": place_schedule["window_frac"],
            "place_hold": place_schedule["hold_frac"],
            "pick_cost": pick_cost,
            "place_cost": place_cost,
            "total_cost": total_cost,
            "reward": reward,
            "pick_min_dist": pick_out["min_dist"],
            "place_min_dist": place_out["min_dist"],
            "pick_arm_energy": pick_out["arm_energy"],
            "place_arm_energy": place_out["arm_energy"],
            "pick_success": pick_success,
            "place_success": place_success,
        }

        self.logs.append(info)

        return self._get_obs(), float(reward), terminated, truncated, info

    def save_logs(self, path=RL_LOG_FILE):
        if self.logs:
            pd.DataFrame(self.logs).to_csv(path, index=False)
            print(f"[RL] Training log saved to {path}")


# ============================================================
# Build mission geometry from CoppeliaSim
# ============================================================

@dataclass
class MissionSetup:
    home: list
    pick: list
    place: list
    pick_flyby: list
    place_flyby: list
    wps1: list
    wps2: list
    wps3: list
    e1: float
    e2: float
    e3: float


def get_scene_geometry_and_optimize_paths():
    """
    Connects to CoppeliaSim briefly, reads positions, and computes optimized drone paths.
    This does not run the full mission.
    """
    client = RemoteAPIClient()
    sim = client.require("sim")
    sim.setStepping(True)

    h = base.get_handles(sim)

    sim.startSimulation()
    time.sleep(0.2)

    try:
        start = sim.getObjectPosition(h.drone_base, -1)
        home = [
            float(start[0]),
            float(start[1]),
            max(float(start[2]), base.SAFE_FLOOR + 0.05),
        ]

        pick = sim.getObjectPosition(h.pickup_obj, -1)
        place = sim.getObjectPosition(h.drop_loc, -1)

        pick = [float(pick[0]), float(pick[1]), float(pick[2])]
        place = [float(place[0]), float(place[1]), float(place[2])]

        pick_flyby = base.make_dynamic_endpoint(pick)
        place_flyby = base.make_dynamic_endpoint(place)

    finally:
        sim.stopSimulation()
        time.sleep(0.2)

    print("\n[RL SETUP] Key points:")
    print(" home        =", np.round(home, 3))
    print(" pick obj    =", np.round(pick, 3))
    print(" pick flyby  =", np.round(pick_flyby, 3))
    print(" place obj   =", np.round(place, 3))
    print(" place flyby =", np.round(place_flyby, 3))

    wps1, e1 = base.optimize_two_vias(
        home,
        pick_flyby,
        base.DRONE_MASS_EMPTY,
        base.DESIRED_SPEED_EMPTY,
        "RL setup home→pick",
    )

    wps2, e2 = base.optimize_two_vias(
        pick_flyby,
        place_flyby,
        base.DRONE_MASS_EMPTY + base.PAYLOAD_MASS_EST,
        base.DESIRED_SPEED_CARRY,
        "RL setup pick→place",
    )

    wps3, e3 = base.optimize_two_vias(
        place_flyby,
        home,
        base.DRONE_MASS_EMPTY,
        base.DESIRED_SPEED_EMPTY,
        "RL setup place→home",
    )

    return MissionSetup(
        home=home,
        pick=pick,
        place=place,
        pick_flyby=pick_flyby,
        place_flyby=place_flyby,
        wps1=wps1,
        wps2=wps2,
        wps3=wps3,
        e1=e1,
        e2=e2,
        e3=e3,
    )


# ============================================================
# Train RL model
# ============================================================

def train_rl(total_timesteps=6000):
    setup = get_scene_geometry_and_optimize_paths()

    env = UAPScheduleEnv(
        home=setup.home,
        pick=setup.pick,
        place=setup.place,
        wps1=setup.wps1,
        wps2=setup.wps2,
        wps3=setup.wps3,
        e1=setup.e1,
        e2=setup.e2,
        e3=setup.e3,
        randomize=True,
    )

    check_env(env, warn=True)
    env = Monitor(env)

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=50000,
        batch_size=128,
        gamma=0.95,
        tau=0.02,
        train_freq=1,
        gradient_steps=1,
        learning_starts=500,
        ent_coef="auto",
        seed=7,
    )

    print("\n[RL] Training started...")
    model.learn(total_timesteps=total_timesteps)
    model.save(RL_MODEL_PATH)

    # Save logs from underlying env
    raw_env = env.env
    if hasattr(raw_env, "save_logs"):
        raw_env.save_logs(RL_LOG_FILE)

    print(f"\n[RL DONE] Model saved as {RL_MODEL_PATH}.zip")


# ============================================================
# Convert trained RL model output to schedules
# ============================================================

def get_rl_schedules(setup):
    env = UAPScheduleEnv(
        home=setup.home,
        pick=setup.pick,
        place=setup.place,
        wps1=setup.wps1,
        wps2=setup.wps2,
        wps3=setup.wps3,
        e1=setup.e1,
        e2=setup.e2,
        e3=setup.e3,
        randomize=False,
    )

    model = SAC.load(RL_MODEL_PATH)

    obs, _ = env.reset()
    action, _ = model.predict(obs, deterministic=True)

    pick_schedule, place_schedule = env._action_to_schedules(action)

    _, _, _, _, info = env.step(action)

    print("\n[RL POLICY OUTPUT]")
    print("Pick schedule:")
    print(" center_frac =", round(pick_schedule["center_frac"], 4))
    print(" window_frac =", round(pick_schedule["window_frac"], 4))
    print(" hold_frac   =", round(pick_schedule["hold_frac"], 4))

    print("\nPlace schedule:")
    print(" center_frac =", round(place_schedule["center_frac"], 4))
    print(" window_frac =", round(place_schedule["window_frac"], 4))
    print(" hold_frac   =", round(place_schedule["hold_frac"], 4))

    print("\nEstimated RL evaluation:")
    for k, v in info.items():
        if isinstance(v, float):
            print(f" {k}: {v:.5f}")
        else:
            print(f" {k}: {v}")

    pick_schedule["arm_energy"] = float(info["pick_arm_energy"])
    pick_schedule["est_min_dist"] = float(info["pick_min_dist"])
    pick_schedule["cost"] = float(info["pick_cost"])

    place_schedule["arm_energy"] = float(info["place_arm_energy"])
    place_schedule["est_min_dist"] = float(info["place_min_dist"])
    place_schedule["cost"] = float(info["place_cost"])

    return pick_schedule, place_schedule


# ============================================================
# Run final CoppeliaSim mission using trained RL schedules
# ============================================================

def run_final_mission_with_rl():
    setup = get_scene_geometry_and_optimize_paths()
    pick_arm_schedule, place_arm_schedule = get_rl_schedules(setup)

    client = RemoteAPIClient()
    sim = client.require("sim")
    sim.setStepping(True)

    h = base.get_handles(sim)
    recorder = base.DataRecorder()

    spline_info = {"waypoints": [], "energies": []}

    for lab, wps, e in [
        ("home→pick", setup.wps1, setup.e1),
        ("pick→place", setup.wps2, setup.e2),
        ("place→home", setup.wps3, setup.e3),
    ]:
        spline_info["waypoints"].append(wps)
        spline_info["energies"].append({"phase": lab, "energy": e})
        recorder.log_energy(lab, e)

    spline_info["energies"].append(
        {"phase": "RL pick arm", "energy": pick_arm_schedule["arm_energy"]}
    )
    spline_info["energies"].append(
        {"phase": "RL place arm", "energy": place_arm_schedule["arm_energy"]}
    )

    sim.startSimulation()

    try:
        sim.setIntegerSignal("reset_pid", 1)
        sim.setIntegerSignal("soft_gains", 0)
        sim.setFloatSignal("extra_thrust", 0.0)

        sim.setObjectPosition(h.drone_target, -1, setup.home)

        base.set_arm(sim, h, base.JOINT_FOLDED)
        base.set_gripper(sim, h, False)

        base.stabilize(sim, h, recorder, base.STAB_INIT, "INIT")

        print("\n[MISSION] Phase 1: RL dynamic fly-by pick")
        base.fly_spline(
            sim,
            h,
            recorder,
            setup.wps1,
            base.JOINT_FOLDED,
            base.JOINT_CARRY,
            "RL_DYNAMIC_PICK",
            speed=base.DESIRED_SPEED_EMPTY,
            dynamic_event={"kind": "pick", "object_world": setup.pick},
            arm_schedule=pick_arm_schedule,
        )

        base.stabilize(sim, h, recorder, base.STAB_GRAB, "POST_PICK")

        print("[MISSION] Phase 2: RL carry to place + dynamic release")
        base.fly_spline(
            sim,
            h,
            recorder,
            setup.wps2,
            base.JOINT_CARRY,
            base.JOINT_CARRY,
            "RL_CARRY_PLACE",
            speed=base.DESIRED_SPEED_CARRY,
            dynamic_event={"kind": "place", "object_world": setup.place},
            arm_schedule=place_arm_schedule,
        )

        base.stabilize(sim, h, recorder, base.STAB_RELEASE, "POST_RELEASE")

        print("[MISSION] Phase 3: return home folded")
        base.fly_spline(
            sim,
            h,
            recorder,
            setup.wps3,
            base.JOINT_CARRY,
            base.JOINT_FOLDED,
            "RL_RETURN_HOME",
            speed=base.DESIRED_SPEED_EMPTY,
            dynamic_event=None,
            arm_schedule=None,
        )

        base.set_gripper(sim, h, False)
        base.stabilize(sim, h, recorder, base.STAB_FINAL, "FINAL")

    finally:
        sim.stopSimulation()

        if recorder.rows:
            df = pd.DataFrame(recorder.rows)

            out_csv = "uap_rl_final_data.csv"
            out_plot = "uap_rl_final_report.png"

            old_log = base.LOG_FILE
            old_plot = base.PLOT_FILE

            # Save CSV
            df.to_csv(out_csv, index=False)
            print(f"\nSaved RL CSV: {out_csv}")

            # Temporarily redirect plot filename if your base script uses global PLOT_FILE
            base.PLOT_FILE = out_plot
            base.plot_results(df, spline_info)
            base.PLOT_FILE = old_plot

            print(f"Saved RL plots: {out_plot}, spline_z_profiles.png, spline_paths_energy.png")

        else:
            warnings.warn("No data recorded.")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "run"],
        help="train = train RL model, run = run final mission using trained RL model",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=6000,
        help="RL training timesteps",
    )

    args = parser.parse_args()

    if args.mode == "train":
        train_rl(total_timesteps=args.timesteps)

    elif args.mode == "run":
        run_final_mission_with_rl()


if __name__ == "__main__":
    main()