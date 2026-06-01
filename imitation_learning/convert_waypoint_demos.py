"""
convert_waypoint_demos.py
Converts waypoint trajectory CSV from previous quadruped project
into observation-action pairs for ANYmal D behavior cloning.
"""
import os
import json
import numpy as np
import pandas as pd
import mujoco

def find_scene_xml():
    seeds = [os.path.dirname(os.path.abspath(__file__)), os.getcwd()]
    for seed in seeds:
        d = seed
        for _ in range(6):
            cand = os.path.join(d, "anybotics_anymal_d", "scene.xml")
            if os.path.exists(cand):
                return cand
            d = os.path.dirname(d)
    raise FileNotFoundError("scene.xml not found")

def get_obs(data):
    return np.concatenate([data.qpos.copy(), data.qvel.copy()])

def velocity_cmd_to_joint_action(model, data, vx_cmd, vy_cmd, wz_cmd, nominal):
    """
    Convert velocity commands from waypoint controller into
    joint position targets for ANYmal D.
    Uses nominal pose + velocity-proportional offset.
    """
    scale = 0.3
    action = nominal.copy()
    # Forward velocity -> extend front legs slightly
    action[0]  += vx_cmd * scale   # LF_HAA
    action[3]  += vx_cmd * scale   # RF_HAA
    action[6]  -= vx_cmd * scale   # LH_HAA
    action[9]  -= vx_cmd * scale   # RH_HAA
    # Lateral velocity -> adjust HFE joints
    action[1]  += vy_cmd * scale
    action[4]  -= vy_cmd * scale
    action[7]  += vy_cmd * scale
    action[10] -= vy_cmd * scale
    # Yaw -> differential HAA
    action[0]  += wz_cmd * scale * 0.5
    action[3]  -= wz_cmd * scale * 0.5
    action[6]  += wz_cmd * scale * 0.5
    action[9]  -= wz_cmd * scale * 0.5
    return np.clip(action, -1.0, 1.0)

def convert_demos(csv_path, output_path, n_repeat=5):
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Rows: {len(df)}, Columns: {list(df.columns)}")

    xml_path = find_scene_xml()
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    nominal = model.key_qpos[0][7:19].copy()

    all_obs, all_acts = [], []

    for repeat in range(n_repeat):
        mujoco.mj_resetDataKeyframe(model, data, 0)
        noise_scale = 0.02 * repeat

        for _, row in df.iterrows():
            vx = float(row.get("vx_cmd_eff", 0.0))
            vy = float(row.get("vy_cmd_eff", 0.0))
            wz = float(row.get("wz_cmd_eff", 0.0))

            obs = get_obs(data)
            action = velocity_cmd_to_joint_action(model, data, vx, vy, wz, nominal)
            action += np.random.randn(12) * noise_scale

            all_obs.append(obs.astype(np.float32))
            all_acts.append(action.astype(np.float32))

            data.ctrl[:] = action
            mujoco.mj_step(model, data)

        print(f"  Repeat {repeat+1}/{n_repeat} done ({len(df)} steps)")

    observations = np.array(all_obs, dtype=np.float32)
    actions = np.array(all_acts, dtype=np.float32)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(
        output_path,
        observations=observations,
        actions=actions,
        metadata=json.dumps({
            "obs_dim": int(observations.shape[1]),
            "act_dim": int(actions.shape[1]),
            "n_samples": int(observations.shape[0]),
            "source": "waypoint_walk_demo_csv",
            "original_repo": "carloAdr1/quadruped-optimal-control-waypoints",
        })
    )
    print(f"Saved: {output_path}")
    print(f"  obs: {observations.shape}, acts: {actions.shape}")
    return observations, actions

if __name__ == "__main__":
    convert_demos(
        csv_path="datasets/waypoint_walk_demo.csv",
        output_path="datasets/anymal_d_demos.npz",
        n_repeat=5,
    )
