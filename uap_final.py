"""
UGP Dynamic Pick-and-Place for CoppeliaSim 4.9
Quadcopter + PhantomX Pincher

FINAL VERSION:
1. Computes cubic splines with exactly 2 optimized via-points for:
   home -> pick, pick -> place, place -> home.
2. Optimizes drone via-points using a rotor-energy proxy.
3. Optimizes arm motion for pick and place using a torque-based energy proxy.
4. Executes dynamic fly-by pick/place:
   - drone follows optimized spline,
   - arm opens/reaches only during optimized event window,
   - pseudo-pick/release happens after the true minimum gripper-object distance
     has been passed, not at first threshold crossing.
5. Saves CSV and plots.
"""

import math
import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import CubicSpline
from scipy.optimize import differential_evolution, minimize
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

SAFE_FLOOR = 0.35
PEAK_HEIGHT = 2.00
DESIRED_SPEED_EMPTY = 0.3
DESIRED_SPEED_CARRY = 0.2
LOG_FILE = "uap_final_data.csv"
PLOT_FILE = "uap_final_report.png"

L1 = 0.10
L2 = 0.10
L3 = 0.10
L4 = 0.08
ARM_REACH = L2 + L3 + L4

JOINT_FOLDED = np.array([0.00, -np.pi / 2, np.pi, -np.pi])
JOINT_PRE_REACH = np.array([0.00, -0.25, 0.70, 0.00])
JOINT_CARRY = np.array([0.00, -0.25, 0.70, 0.00])

GRIPPER_OPEN_ANGLE = 0.50
GRIPPER_CLOSE_ANGLE = -0.30

HOVER_Z_OFFSET = 0.65
FLYBY_Z_OFFSET = 0.46
REACH_THRESHOLD = 0.18

STAB_INIT = 1.0
STAB_GRAB = 0.35
STAB_RELEASE = 0.35
STAB_FINAL = 0.5

RHO = 1.225
ROTOR_RADIUS = 0.12
ROTOR_AREA = math.pi * ROTOR_RADIUS**2
N_ROTORS = 4
DRONE_MASS_EMPTY = 0.80
PAYLOAD_MASS_EST = 0.08
G = 9.81

W_TIME = 0.10
W_ACC_XY = 0.12
W_JERK = 0.025
W_FLOOR = 5000.0
W_HEIGHT = 0.04
W_CURVE = 0.01

W_ARM_TORQUE = 1.0
W_GRASP = 2500.0
W_ARM_MOTION = 0.03
W_ARM_TIME = 0.02

JOINT_INERTIA = np.array([0.015, 0.010, 0.008, 0.004])
JOINT_DAMPING = np.array([0.020, 0.018, 0.015, 0.010])
JOINT_GRAVITY_COEFF = np.array([0.000, 0.080, 0.050, 0.020])

np.random.seed(7)


def smoothstep(s: float) -> float:
    s = float(np.clip(s, 0.0, 1.0))
    return s * s * (3.0 - 2.0 * s)


def quintic(s: float) -> float:
    s = float(np.clip(s, 0.0, 1.0))
    return s**3 * (10.0 - 15.0 * s + 6.0 * s**2)


def build_cubic_spline(waypoints, n_pts=300):
    waypoints = np.asarray(waypoints, dtype=float)
    d = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    u = np.r_[0.0, np.cumsum(d)]
    if u[-1] < 1e-9:
        u = np.linspace(0, 1, len(waypoints))
    else:
        u = u / u[-1]
    t = np.linspace(0, 1, n_pts)
    xyz = np.zeros((n_pts, 3))
    for k in range(3):
        cs = CubicSpline(u, waypoints[:, k], bc_type="natural")
        xyz[:, k] = cs(t)
    return t, xyz


def rotor_power_from_total_thrust(total_thrust):
    total_thrust = np.maximum(total_thrust, 1e-6)
    t_each = total_thrust / N_ROTORS
    return N_ROTORS * (t_each ** 1.5) / math.sqrt(2.0 * RHO * ROTOR_AREA)


def estimate_segment_energy(waypoints, mass, speed, n_pts=350):
    _, xyz = build_cubic_spline(waypoints, n_pts=n_pts)
    path_len = float(np.sum(np.linalg.norm(np.diff(xyz, axis=0), axis=1)))
    tf = max(path_len / max(speed, 1e-3), 0.5)
    dt = tf / (n_pts - 1)
    vel = np.gradient(xyz, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)
    jerk = np.gradient(acc, dt, axis=0)
    total_thrust = mass * np.maximum(G + acc[:, 2], 0.1)
    p_rot = rotor_power_from_total_thrust(total_thrust)
    xy_acc_cost = W_ACC_XY * np.sum(acc[:, :2] ** 2, axis=1)
    jerk_cost = W_JERK * np.sum(jerk ** 2, axis=1)
    height_cost = W_HEIGHT * xyz[:, 2]
    floor_violation = np.maximum(SAFE_FLOOR - xyz[:, 2], 0.0)
    ceiling_violation = np.maximum(xyz[:, 2] - PEAK_HEIGHT, 0.0)
    violation_cost = W_FLOOR * (floor_violation**2 + ceiling_violation**2)
    straight = np.linalg.norm(np.asarray(waypoints[-1]) - np.asarray(waypoints[0]))
    curve_cost = W_CURVE * max(path_len - straight, 0.0)
    integrand = p_rot + xy_acc_cost + jerk_cost + height_cost + violation_cost
    return float(np.trapz(integrand, dx=dt) + W_TIME * tf + curve_cost)


def optimize_two_vias(p0, p3, mass=DRONE_MASS_EMPTY, speed=DESIRED_SPEED_EMPTY, label="segment"):
    p0 = np.asarray(p0, dtype=float)
    p3 = np.asarray(p3, dtype=float)
    v1_guess = p0 + (p3 - p0) / 3.0
    v2_guess = p0 + 2.0 * (p3 - p0) / 3.0
    base = np.r_[v1_guess, v2_guess]
    span = max(np.linalg.norm(p3[:2] - p0[:2]), 0.5)
    xy_pad = max(0.35, 0.35 * span)
    min_x = min(p0[0], p3[0]) - xy_pad
    max_x = max(p0[0], p3[0]) + xy_pad
    min_y = min(p0[1], p3[1]) - xy_pad
    max_y = max(p0[1], p3[1]) + xy_pad
    bounds = [
        (min_x, max_x), (min_y, max_y), (SAFE_FLOOR + 0.05, PEAK_HEIGHT),
        (min_x, max_x), (min_y, max_y), (SAFE_FLOOR + 0.05, PEAK_HEIGHT),
    ]
    def obj(x):
        reg = 0.015 * float(np.sum((np.asarray(x) - base) ** 2))
        wps = [p0, x[:3], x[3:], p3]
        return estimate_segment_energy(wps, mass=mass, speed=speed) + reg
    print(f"\n[OPT] {label}: differential evolution coarse search")
    de = differential_evolution(obj, bounds=bounds, maxiter=45, popsize=9, polish=False, tol=1e-3, workers=1, seed=7, updating="immediate")
    print(f"[OPT] {label}: L-BFGS-B refinement")
    res = minimize(obj, de.x, method="L-BFGS-B", bounds=bounds, options={"maxiter": 250, "ftol": 1e-8})
    x = res.x
    wps = [p0.tolist(), x[:3].tolist(), x[3:].tolist(), p3.tolist()]
    e = estimate_segment_energy(wps, mass=mass, speed=speed)
    print(f"[OPT DONE] {label}: E={e:.3f}, v1={np.round(x[:3], 3)}, v2={np.round(x[3:], 3)}")
    return wps, e


def ik_arm(target_local):
    x, y, z = map(float, target_local)
    q1 = math.atan2(y, x)
    r = math.hypot(x, y)
    rw = r - L4
    zw = z - L1
    d = math.hypot(rw, zw)
    d = np.clip(d, 1e-5, L2 + L3 - 1e-4)
    c3 = np.clip((d * d - L2 * L2 - L3 * L3) / (2.0 * L2 * L3), -1.0, 1.0)
    q3 = -math.acos(c3)
    alpha = math.atan2(zw, rw)
    beta = math.atan2(L3 * math.sin(-q3), L2 + L3 * math.cos(-q3))
    q2 = alpha - beta
    q4 = np.clip(-(q2 + q3), -1.5, 1.5)
    return np.array([q1, q2, q3, q4], dtype=float)


def forward_tip_local(q):
    q1, q2, q3, q4 = np.asarray(q, dtype=float)
    r = L2 * math.cos(q2) + L3 * math.cos(q2 + q3) + L4 * math.cos(q2 + q3 + q4)
    z = L1 + L2 * math.sin(q2) + L3 * math.sin(q2 + q3) + L4 * math.sin(q2 + q3 + q4)
    return np.array([r * math.cos(q1), r * math.sin(q1), z], dtype=float)


def estimate_arm_energy_torque(q_traj, tf):
    q_traj = np.asarray(q_traj, dtype=float)
    if len(q_traj) < 3:
        return 0.0
    dt = tf / max(len(q_traj) - 1, 1)
    qd = np.gradient(q_traj, dt, axis=0)
    qdd = np.gradient(qd, dt, axis=0)
    gravity = np.zeros_like(q_traj)
    gravity[:, 1] = JOINT_GRAVITY_COEFF[1] * np.sin(q_traj[:, 1])
    gravity[:, 2] = JOINT_GRAVITY_COEFF[2] * np.sin(q_traj[:, 1] + q_traj[:, 2])
    gravity[:, 3] = JOINT_GRAVITY_COEFF[3] * np.sin(q_traj[:, 1] + q_traj[:, 2] + q_traj[:, 3])
    tau = JOINT_INERTIA * qdd + JOINT_DAMPING * qd + gravity
    power = np.sum(np.abs(tau * qd), axis=1)
    return float(np.trapz(power, dx=dt))


def approx_object_local_from_path(drone_pos, object_world):
    return np.asarray(object_world, dtype=float) - np.asarray(drone_pos, dtype=float)


def optimize_arm_event_schedule(waypoints, object_world, q_start, q_end, speed, label="pick/place"):
    n = 140
    _, xyz = build_cubic_spline(waypoints, n)
    path_len = float(np.sum(np.linalg.norm(np.diff(xyz, axis=0), axis=1)))
    tf = max(path_len / max(speed, 1e-3), 1.0)
    q_start = np.asarray(q_start, dtype=float)
    q_end = np.asarray(q_end, dtype=float)
    obj = np.asarray(object_world, dtype=float)
    xy_dists = np.linalg.norm(xyz[:, :2] - obj[:2], axis=1)
    event_guess = int(np.argmin(xy_dists)) / max(n - 1, 1)
    def build_q_traj(params):
        center_frac, window_frac, hold_frac = params
        center_frac = float(np.clip(center_frac, 0.05, 0.95))
        window_frac = float(np.clip(window_frac, 0.06, 0.55))
        hold_frac = float(np.clip(hold_frac, 0.00, 0.18))
        t = np.linspace(0, 1, n)
        q_traj = np.zeros((n, 4))
        tip_dists = np.zeros(n)
        half_window = window_frac / 2.0
        half_hold = hold_frac / 2.0
        for k, tk in enumerate(t):
            q_nom = q_start + (q_end - q_start) * quintic(tk)
            local = approx_object_local_from_path(xyz[k], obj)
            q_ik = ik_arm(local)
            d_event = abs(tk - center_frac)
            if d_event <= half_hold:
                blend = 1.0
            elif d_event <= half_window:
                blend_raw = 1.0 - (d_event - half_hold) / max(half_window - half_hold, 1e-6)
                blend = quintic(blend_raw)
            else:
                blend = 0.0
            q = (1.0 - blend) * q_nom + blend * q_ik
            q_traj[k] = q
            tip_world = xyz[k] + forward_tip_local(q)
            tip_dists[k] = float(np.linalg.norm(tip_world - obj))
        return q_traj, tip_dists
    def obj_fun(params):
        q_traj, tip_dists = build_q_traj(params)
        e_arm = estimate_arm_energy_torque(q_traj, tf)
        min_dist = float(np.min(tip_dists))
        motion_mag = float(np.trapz(np.sum((q_traj - q_start) ** 2, axis=1), dx=tf / max(n - 1, 1)))
        center_frac, window_frac, hold_frac = params
        timing_reg = 0.20 * (center_frac - event_guess) ** 2
        window_reg = W_ARM_TIME * window_frac
        return W_ARM_TORQUE * e_arm + W_GRASP * min_dist**2 + W_ARM_MOTION * motion_mag + timing_reg + window_reg
    bounds = [(max(0.05, event_guess - 0.25), min(0.95, event_guess + 0.25)), (0.08, 0.50), (0.00, 0.12)]
    print(f"\n[ARM OPT] {label}: torque-energy + grasp-distance optimization")
    de = differential_evolution(obj_fun, bounds=bounds, maxiter=30, popsize=8, polish=False, tol=1e-3, workers=1, seed=11, updating="immediate")
    res = minimize(obj_fun, de.x, method="L-BFGS-B", bounds=bounds, options={"maxiter": 150, "ftol": 1e-8})
    center_frac, window_frac, hold_frac = res.x
    q_traj, tip_dists = build_q_traj(res.x)
    e_arm = estimate_arm_energy_torque(q_traj, tf)
    min_dist = float(np.min(tip_dists))
    print(f"[ARM OPT DONE] {label}: center={center_frac:.3f}, window={window_frac:.3f}, hold={hold_frac:.3f}, E_arm={e_arm:.5f}, est_min_dist={min_dist:.4f}, cost={res.fun:.4f}")
    return {"center_frac": float(center_frac), "window_frac": float(window_frac), "hold_frac": float(hold_frac), "arm_energy": float(e_arm), "est_min_dist": float(min_dist), "cost": float(res.fun)}


@dataclass
class Handles:
    drone_base: int
    drone_target: int
    arm_base: int
    joints: list
    pickup_obj: int
    drop_loc: int
    gripper_link: int
    gripper_center: int | None
    gripper_close: int | None


def try_get(sim, path):
    try:
        return sim.getObject(path)
    except Exception:
        return None


def get_handles(sim):
    arm_root = "/Quadcopter/base/PhantomXPincher"
    joints = [sim.getObject(f"{arm_root}/joint"), sim.getObject(f"{arm_root}/joint/link/joint"), sim.getObject(f"{arm_root}/joint/link/joint/link/joint"), sim.getObject(f"{arm_root}/joint/link/joint/link/joint/link/joint")]
    gp_base = f"{arm_root}/joint/link/joint/link/joint/link/joint/link/gripperCenter_joint"
    gripper_center = try_get(sim, gp_base)
    gripper_close = try_get(sim, gp_base + "/fingerLeft/gripperClose_joint")
    try:
        gripper_link = sim.getObject(f"{gp_base}/fingerLeft")
    except Exception:
        gripper_link = sim.getObjectChild(joints[3], 0)
    return Handles(sim.getObject("/Quadcopter/base"), sim.getObject("/target"), sim.getObject(arm_root), joints, sim.getObject("/PickupObject"), sim.getObject("/DropLocation"), gripper_link, gripper_center, gripper_close)


def set_arm(sim, h: Handles, q):
    for jh, val in zip(h.joints, q):
        sim.setJointTargetPosition(jh, float(val))


def set_gripper(sim, h: Handles, open_gripper: bool):
    angle = GRIPPER_OPEN_ANGLE if open_gripper else GRIPPER_CLOSE_ANGLE
    for jh in [h.gripper_center, h.gripper_close]:
        if jh is not None:
            try:
                sim.setJointTargetPosition(jh, float(angle))
            except Exception:
                pass


def world_to_arm_local(sim, h: Handles, p_world):
    mat = sim.getObjectMatrix(h.arm_base, -1)
    R = np.array([[mat[0], mat[1], mat[2]], [mat[4], mat[5], mat[6]], [mat[8], mat[9], mat[10]]], dtype=float)
    T = np.array([mat[3], mat[7], mat[11]], dtype=float)
    return R.T @ (np.asarray(p_world, dtype=float) - T)


def step_and_record(sim, h, recorder, phase, sleep=True):
    sim.step()
    recorder.sample(sim, h, phase)
    if sleep:
        time.sleep(sim.getSimulationTimeStep())


def stabilize(sim, h, recorder, duration, phase):
    n = max(1, int(duration / sim.getSimulationTimeStep()))
    for _ in range(n):
        step_and_record(sim, h, recorder, phase + "_STAB")


class DataRecorder:
    def __init__(self):
        self.rows = []
        self.energies = []
    @staticmethod
    def sig(sim, name, default=0.0):
        try:
            v = sim.getFloatSignal(name)
            return default if v is None else float(v)
        except Exception:
            return default
    def sample(self, sim, h: Handles, phase):
        row = {"time": sim.getSimulationTime(), "phase": phase}
        for s in ["drone_x", "drone_y", "drone_z", "vel_x", "vel_y", "vel_z", "drone_roll", "drone_pitch", "drone_yaw", "Ixx", "Iyy", "Izz", "Ixy", "Ixz", "Iyz", "thrust", "alphaCorr", "betaCorr", "rotCorr"]:
            row[s] = self.sig(sim, s)
        for i in range(1, 5):
            row[f"rotor{i}_force"] = self.sig(sim, f"rotor{i}_force")
            row[f"rotor{i}_vel"] = self.sig(sim, f"rotor{i}_vel")
        for i, jh in enumerate(h.joints, start=1):
            try:
                row[f"arm_j{i}_pos"] = sim.getJointPosition(jh)
            except Exception:
                row[f"arm_j{i}_pos"] = np.nan
        self.rows.append(row)
    def log_energy(self, phase, energy):
        self.energies.append({"phase": phase, "energy": float(energy)})


def plot_results(df, spline_info):
    fig = plt.figure(figsize=(22, 15), facecolor="#f0f2f5")
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.32)
    t = df["time"]
    colors = ["#2196F3", "#4CAF50", "#E91E63"]
    labels = ["home→pick", "pick→place", "place→home"]
    ax1 = fig.add_subplot(gs[0, 0]); ax1.plot(t, df["drone_z"], label="Z actual"); ax1.axhline(SAFE_FLOOR, ls="--", lw=0.9, label="safe floor"); ax1.set_title("Altitude"); ax1.legend(fontsize=7)
    ax2 = fig.add_subplot(gs[0, 1]); ax2.plot(df["drone_x"], df["drone_y"], label="actual")
    for wps, c, lab in zip(spline_info["waypoints"], colors, labels):
        _, xyz = build_cubic_spline(wps, 300); ax2.plot(xyz[:, 0], xyz[:, 1], "--", color=c, label=lab)
        for vp in wps[1:3]: ax2.plot(vp[0], vp[1], "x", color=c, ms=8)
    ax2.set_title("XY Path + optimized via-points"); ax2.legend(fontsize=7)
    ax3 = fig.add_subplot(gs[0, 2])
    for i in range(1, 5): ax3.plot(t, np.degrees(df[f"arm_j{i}_pos"].fillna(0)), label=f"J{i}")
    ax3.set_title("Arm joints"); ax3.legend(fontsize=7)
    ax4 = fig.add_subplot(gs[1, 0])
    for c in ["x", "y", "z"]: ax4.plot(t, df[f"vel_{c}"], label=f"V{c}")
    ax4.set_title("Velocity"); ax4.legend(fontsize=7)
    ax5 = fig.add_subplot(gs[1, 1]); ax5.plot(t, np.degrees(df["drone_roll"]), label="roll signal"); ax5.plot(t, np.degrees(df["drone_pitch"]), label="pitch signal"); ax5.set_title("Attitude signals"); ax5.legend(fontsize=7)
    ax6 = fig.add_subplot(gs[1, 2])
    for i in range(1, 5): ax6.plot(t, df[f"rotor{i}_force"], label=f"R{i}")
    ax6.set_title("Rotor forces / commands"); ax6.legend(fontsize=7)
    ax7 = fig.add_subplot(gs[2, 0]); ax7.plot(t, df["drone_x"]); ax7.set_title("X tracking")
    ax8 = fig.add_subplot(gs[2, 1]); ax8.plot(t, df["drone_y"]); ax8.set_title("Y tracking")
    ax9 = fig.add_subplot(gs[2, 2])
    for m in ["Ixx", "Iyy", "Izz", "Ixy", "Ixz", "Iyz"]: ax9.plot(t, df[m], label=m)
    ax9.set_title("Dynamic MOI"); ax9.legend(fontsize=7)
    fig.suptitle("UAP Simulation Report", fontsize=14, y=1.01); fig.savefig(PLOT_FILE, bbox_inches="tight"); plt.show()
    fig2, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor="#f0f2f5")
    for ax, wps, c, lab in zip(axes, spline_info["waypoints"], colors, labels):
        _, xyz = build_cubic_spline(wps, 300); u = np.linspace(0, 1, len(xyz)); ax.plot(u, xyz[:, 2], color=c, lw=2.5)
        for vp in wps[1:3]: ax.axhline(vp[2], ls=":", lw=0.8)
        ax.axhline(SAFE_FLOOR, ls="--", lw=0.8); ax.set_title(f"Planned Z — {lab}"); ax.set_xlabel("normalized spline parameter"); ax.set_ylabel("Z")
        z_vals = xyz[:, 2]; ax.set_ylim(max(0, z_vals.min() - 0.08), z_vals.max() + 0.08)
    fig2.tight_layout(); fig2.savefig("spline_z_profiles.png", bbox_inches="tight"); plt.show()
    fig3 = plt.figure(figsize=(16, 6), facecolor="#f0f2f5"); ax3d = fig3.add_subplot(1, 2, 1, projection="3d")
    for wps, c, lab in zip(spline_info["waypoints"], colors, labels):
        _, xyz = build_cubic_spline(wps, 300); ax3d.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=c, lw=2, label=lab)
        for vp in wps[1:3]: ax3d.scatter(vp[0], vp[1], vp[2], color=c, marker="x", s=50)
    ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z"); ax3d.set_title("3D spline paths"); ax3d.legend(fontsize=7)
    ax_e = fig3.add_subplot(1, 2, 2); names = [e["phase"] for e in spline_info["energies"]]; vals = [e["energy"] for e in spline_info["energies"]]; bars = ax_e.bar(names, vals)
    ax_e.set_ylabel("Energy proxy"); ax_e.set_title("Estimated energy per segment")
    for bar, val in zip(bars, vals): ax_e.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01, f"{val:.2f}", ha="center", fontsize=8)
    fig3.tight_layout(); fig3.savefig("spline_paths_energy.png", bbox_inches="tight"); plt.show()


def fly_spline(
    sim,
    h,
    recorder,
    waypoints,
    q_start,
    q_end,
    phase,
    speed=DESIRED_SPEED_EMPTY,
    dynamic_event=None,
    arm_schedule=None,
):
    sim_dt = sim.getSimulationTimeStep()

    _, xyz_tmp = build_cubic_spline(waypoints, 350)
    path_len = float(np.sum(np.linalg.norm(np.diff(xyz_tmp, axis=0), axis=1)))
    tf = max(path_len / max(speed, 1e-3), 1.0)

    n = max(60, int(tf / sim_dt))
    _, xyz = build_cubic_spline(waypoints, n)

    q_start = np.asarray(q_start, dtype=float)
    q_end = np.asarray(q_end, dtype=float)

    did_action = False

    if dynamic_event is not None:
        obj = np.asarray(dynamic_event["object_world"], dtype=float)

        event_idx = int(np.argmin(np.linalg.norm(xyz[:, :2] - obj[:2], axis=1)))
        event_idx = int(np.clip(event_idx, 5, n - 6))

        if arm_schedule is None:
            arm_schedule = {
                "center_frac": event_idx / max(n - 1, 1),
                "window_frac": 0.25,
                "hold_frac": 0.05,
            }

        center_frac = arm_schedule["center_frac"]
        window_frac = arm_schedule["window_frac"]
        hold_frac = arm_schedule["hold_frac"]

        half_window = window_frac / 2.0
        half_hold = hold_frac / 2.0

    else:
        obj = None
        event_idx = None
        center_frac = 0.5
        half_window = 0.0
        half_hold = 0.0

    for i, pt in enumerate(xyz):
        frac = i / max(n - 1, 1)

        sim.setObjectPosition(
            h.drone_target,
            -1,
            [float(pt[0]), float(pt[1]), float(pt[2])],
        )

        q_nominal = q_start + (q_end - q_start) * quintic(frac)
        q_cmd = q_nominal.copy()
        phase_now = phase

        if dynamic_event is not None:
            d_event = abs(frac - center_frac)

            if d_event <= half_window:
                local = world_to_arm_local(sim, h, obj)
                q_event = ik_arm(local)

                if d_event <= half_hold:
                    blend = 1.0
                else:
                    blend_raw = 1.0 - (d_event - half_hold) / max(
                        half_window - half_hold,
                        1e-6,
                    )
                    blend = quintic(blend_raw)

                q_cmd = (1.0 - blend) * q_nominal + blend * q_event

                if not did_action:
                    set_gripper(sim, h, True)
            else:
                if not did_action:
                    set_gripper(sim, h, False)

        set_arm(sim, h, q_cmd)

        if dynamic_event is not None and not did_action and i >= event_idx:
            if dynamic_event["kind"] == "pick":
                print(">>> ATTACHING OBJECT at event_idx <<<")

                set_gripper(sim, h, False)

                try:
                    sim.setObjectInt32Param(
                        h.pickup_obj,
                        sim.shapeintparam_static,
                        1,
                    )
                except Exception:
                    pass

                sim.setObjectParent(h.pickup_obj, h.gripper_link, True)

                sim.setIntegerSignal("reset_pid", 1)
                sim.setIntegerSignal("soft_gains", 1)
                sim.setFloatSignal("extra_thrust", 0.15)

                did_action = True
                phase_now = phase + "_ATTACH"

            elif dynamic_event["kind"] == "place":
                print(">>> RELEASING OBJECT at event_idx <<<")

                set_gripper(sim, h, True)

                try:
                    sim.setObjectInt32Param(
                        h.pickup_obj,
                        sim.shapeintparam_static,
                        0,
                    )
                except Exception:
                    pass

                sim.setObjectParent(h.pickup_obj, -1, True)

                sim.setIntegerSignal("reset_pid", 1)
                sim.setIntegerSignal("soft_gains", 1)
                sim.setFloatSignal("extra_thrust", 0.0)

                did_action = True
                phase_now = phase + "_RELEASE"

        step_and_record(sim, h, recorder, phase_now)

    if dynamic_event is not None:
        set_gripper(sim, h, False)

    sim.setIntegerSignal("soft_gains", 0)


def make_dynamic_endpoint(obj_pos, current_z=None):
    return [float(obj_pos[0]), float(obj_pos[1] - 0.17), float(max(SAFE_FLOOR, obj_pos[2] + FLYBY_Z_OFFSET - 0.05))]


def main():
    client = RemoteAPIClient()
    sim = client.require("sim")
    sim.setStepping(True)
    h = get_handles(sim)
    recorder = DataRecorder()
    sim.startSimulation()
    try:
        sim.setIntegerSignal("reset_pid", 1)
        sim.setIntegerSignal("soft_gains", 0)
        sim.setFloatSignal("extra_thrust", 0.0)
        start = sim.getObjectPosition(h.drone_base, -1)
        home = [float(start[0]), float(start[1]), max(float(start[2]), SAFE_FLOOR + 0.05)]
        sim.setObjectPosition(h.drone_target, -1, home)
        set_arm(sim, h, JOINT_FOLDED)
        set_gripper(sim, h, False)
        stabilize(sim, h, recorder, STAB_INIT, "INIT")
        pick = sim.getObjectPosition(h.pickup_obj, -1)
        place = sim.getObjectPosition(h.drop_loc, -1)
        pick_flyby = make_dynamic_endpoint(pick)
        place_flyby = make_dynamic_endpoint(place)
        print("\nKey points:")
        print(" home      =", np.round(home, 3))
        print(" pick obj  =", np.round(pick, 3), " pick fly-by =", np.round(pick_flyby, 3))
        print(" place obj =", np.round(place, 3), " place fly-by=", np.round(place_flyby, 3))
        spline_info = {"waypoints": [], "energies": []}
        wps1, e1 = optimize_two_vias(home, pick_flyby, DRONE_MASS_EMPTY, DESIRED_SPEED_EMPTY, "home→pick")
        wps2, e2 = optimize_two_vias(pick_flyby, place_flyby, DRONE_MASS_EMPTY + PAYLOAD_MASS_EST, DESIRED_SPEED_CARRY, "pick→place")
        wps3, e3 = optimize_two_vias(place_flyby, home, DRONE_MASS_EMPTY, DESIRED_SPEED_EMPTY, "place→home")
        pick_arm_schedule = optimize_arm_event_schedule(wps1, pick, JOINT_FOLDED, JOINT_CARRY, DESIRED_SPEED_EMPTY, "PICK arm")
        place_arm_schedule = optimize_arm_event_schedule(wps2, place, JOINT_CARRY, JOINT_CARRY, DESIRED_SPEED_CARRY, "PLACE arm")
        for lab, wps, e in [("home→pick", wps1, e1), ("pick→place", wps2, e2), ("place→home", wps3, e3)]:
            spline_info["waypoints"].append(wps)
            spline_info["energies"].append({"phase": lab, "energy": e})
            recorder.log_energy(lab, e)
        spline_info["energies"].append({"phase": "pick arm", "energy": pick_arm_schedule["arm_energy"]})
        spline_info["energies"].append({"phase": "place arm", "energy": place_arm_schedule["arm_energy"]})
        print("\nTotal optimized drone transit energy proxy:", round(e1 + e2 + e3, 3))
        print("Total optimized arm torque-energy proxy:", round(pick_arm_schedule["arm_energy"] + place_arm_schedule["arm_energy"], 5))
        print("\nPhase 1: dynamic fly-by pick with optimized arm schedule")
        fly_spline(sim, h, recorder, wps1, JOINT_FOLDED, JOINT_CARRY, "DYNAMIC_PICK", speed=DESIRED_SPEED_EMPTY, dynamic_event={"kind": "pick", "object_world": pick}, arm_schedule=pick_arm_schedule)
        stabilize(sim, h, recorder, STAB_GRAB, "POST_PICK")
        print("Phase 2: carry to place + optimized dynamic release")
        fly_spline(sim, h, recorder, wps2, JOINT_CARRY, JOINT_CARRY, "CARRY_PLACE", speed=DESIRED_SPEED_CARRY, dynamic_event={"kind": "place", "object_world": place}, arm_schedule=place_arm_schedule)
        stabilize(sim, h, recorder, STAB_RELEASE, "POST_RELEASE")
        print("Phase 3: return home folded")
        fly_spline(sim, h, recorder, wps3, JOINT_CARRY, JOINT_FOLDED, "RETURN_HOME", speed=DESIRED_SPEED_EMPTY, dynamic_event=None, arm_schedule=None)
        set_gripper(sim, h, False)
        stabilize(sim, h, recorder, STAB_FINAL, "FINAL")
    finally:
        sim.stopSimulation()
        if recorder.rows:
            df = pd.DataFrame(recorder.rows)
            df.to_csv(LOG_FILE, index=False)
            print(f"\nSaved CSV: {LOG_FILE}")
            plot_results(df, spline_info)
            print(f"Saved plots: {PLOT_FILE}, spline_z_profiles.png, spline_paths_energy.png")
        else:
            warnings.warn("No data recorded.")


if __name__ == "__main__":
    main()
