# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Reward functions for the stacking task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ============================================================================
# Reaching Rewards
# ============================================================================

def gripper_distance_to_cube(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
) -> torch.Tensor:
    """Reward for gripper approaching the target cube.
    
    Returns exponentially decreasing reward based on distance.
    Closer distance = higher reward.
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    cube: RigidObject = env.scene[cube_cfg.name]
    
    ee_pos = ee_frame.data.target_pos_w[:, 0, :]
    cube_pos = cube.data.root_pos_w
    
    distance = torch.norm(cube_pos - ee_pos, dim=1)
    # Exponential reward: closer = higher reward
    # At distance 0: reward = 1.0
    # At distance 0.2m: reward ~= 0.37
    return torch.exp(-5.0 * distance)


def gripper_distance_to_cube_xy_only(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
) -> torch.Tensor:
    """Reward for gripper approaching cube in XY plane only.
    
    Useful for encouraging horizontal alignment before vertical approach.
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    cube: RigidObject = env.scene[cube_cfg.name]
    
    ee_pos = ee_frame.data.target_pos_w[:, 0, :2]  # Only XY
    cube_pos = cube.data.root_pos_w[:, :2]  # Only XY
    
    distance = torch.norm(cube_pos - ee_pos, dim=1)
    return torch.exp(-5.0 * distance)


# ============================================================================
# Grasping Rewards
# ============================================================================

def grasp_bonus(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    distance_threshold: float = 0.06,
) -> torch.Tensor:
    """Bonus reward for successfully grasping a cube.
    
    Returns 1.0 if cube is grasped, 0.0 otherwise.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    cube: RigidObject = env.scene[cube_cfg.name]
    
    # Check distance
    ee_pos = ee_frame.data.target_pos_w[:, 0, :]
    cube_pos = cube.data.root_pos_w
    distance = torch.norm(cube_pos - ee_pos, dim=1)
    
    # Check gripper is closed
    if hasattr(env.cfg, "gripper_joint_names"):
        gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
        gripper_pos = robot.data.joint_pos[:, gripper_joint_ids[0]]
        
        is_closed = torch.abs(
            gripper_pos - torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device)
        ) > env.cfg.gripper_threshold
        
        grasped = torch.logical_and(distance < distance_threshold, is_closed)
        return grasped.float()
    
    return torch.zeros(env.num_envs, device=env.device)


# ============================================================================
# Lifting Rewards
# ============================================================================

def cube_lift_reward(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    target_height: float = 0.15,
) -> torch.Tensor:
    """Reward for lifting cube above table.
    
    Returns value between 0.0 and 1.0 based on height.
    """
    cube: RigidObject = env.scene[cube_cfg.name]
    
    # Height above table (table is at z=0 in env frame)
    cube_height = cube.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    
    # Normalize to [0, 1] range
    # Height 0: reward = 0
    # Height = target_height: reward = 1.0
    # Height > target_height: reward = 1.0 (clamped)
    height_reward = torch.clamp(cube_height / target_height, 0.0, 1.0)
    
    return height_reward


def cube_lift_bonus(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    height_threshold: float = 0.05,
) -> torch.Tensor:
    """Binary bonus for lifting cube above threshold.
    
    Returns 1.0 if lifted, 0.0 otherwise.
    """
    cube: RigidObject = env.scene[cube_cfg.name]
    cube_height = cube.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    
    is_lifted = cube_height > height_threshold
    return is_lifted.float()


# ============================================================================
# Stacking Rewards
# ============================================================================

def cube_stacking_reward(
    env: ManagerBasedRLEnv,
    upper_cube_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    lower_cube_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    xy_threshold: float = 0.05,
    height_diff: float = 0.0468,
) -> torch.Tensor:
    """Reward for stacking upper cube on lower cube.
    
    Combines XY alignment and height alignment rewards.
    """
    upper_cube: RigidObject = env.scene[upper_cube_cfg.name]
    lower_cube: RigidObject = env.scene[lower_cube_cfg.name]
    
    pos_diff = upper_cube.data.root_pos_w - lower_cube.data.root_pos_w
    
    # XY alignment reward (horizontal alignment)
    xy_dist = torch.norm(pos_diff[:, :2], dim=1)
    xy_reward = torch.exp(-10.0 * xy_dist)
    
    # Height alignment reward (vertical alignment)
    height_dist = torch.abs(pos_diff[:, 2] - height_diff)
    height_reward = torch.exp(-50.0 * height_dist)
    
    # Combined reward: both need to be good
    return xy_reward * height_reward


def cube_above_cube_bonus(
    env: ManagerBasedRLEnv,
    upper_cube_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    lower_cube_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    xy_threshold: float = 0.08,
) -> torch.Tensor:
    """Binary bonus for placing upper cube roughly above lower cube.
    
    More lenient than full stacking check.
    """
    upper_cube: RigidObject = env.scene[upper_cube_cfg.name]
    lower_cube: RigidObject = env.scene[lower_cube_cfg.name]
    
    pos_diff = upper_cube.data.root_pos_w - lower_cube.data.root_pos_w
    xy_dist = torch.norm(pos_diff[:, :2], dim=1)
    
    # Check if upper is above lower
    is_above = pos_diff[:, 2] < 0.0  # Upper should have lower z than lower cube
    is_aligned = xy_dist < xy_threshold
    
    return torch.logical_and(is_above, is_aligned).float()


# ============================================================================
# Success Rewards
# ============================================================================

def success_bonus(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
) -> torch.Tensor:
    """Large bonus for successfully stacking all 3 cubes."""
    from .terminations import cubes_stacked
    
    stacked = cubes_stacked(env, robot_cfg, cube_1_cfg, cube_2_cfg, cube_3_cfg)
    return 100.0 * stacked.float()


def success_bonus_2cubes(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
) -> torch.Tensor:
    """Large bonus for successfully stacking 2 cubes."""
    from .terminations import two_cubes_stacked
    
    stacked = two_cubes_stacked(env, robot_cfg, cube_1_cfg, cube_2_cfg)
    return 50.0 * stacked.float()


# ============================================================================
# Penalty Terms
# ============================================================================

def action_rate_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalty for large changes in actions (encourages smooth motion).
    
    Penalizes L2 norm of action differences.
    """
    # Store previous actions for next step
    if not hasattr(env, "_prev_actions"):
        env._prev_actions = env.action_manager.action.clone()
        return torch.zeros(env.num_envs, device=env.device)
    
    action_diff = torch.sum(torch.square(env.action_manager.action - env._prev_actions), dim=1)
    env._prev_actions = env.action_manager.action.clone()
    
    return -0.01 * action_diff


def gripper_open_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Small penalty when gripper is open (to encourage grasping).
    
    Can help agent learn to close gripper more often.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    
    if hasattr(env.cfg, "gripper_joint_names"):
        gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
        gripper_pos = robot.data.joint_pos[:, gripper_joint_ids[0]]
        
        # Penalty if gripper is fully open
        is_open = torch.isclose(
            gripper_pos,
            torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device),
            atol=0.001,
        )
        return -0.1 * is_open.float()
    
    return torch.zeros(env.num_envs, device=env.device)


def cube_orientation_penalty(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
) -> torch.Tensor:
    """Penalty for cube being tilted (not upright).
    
    Encourages stable stacking with cubes remaining upright.
    """
    cube: RigidObject = env.scene[cube_cfg.name]
    
    # Get cube orientation (quaternion)
    quat = cube.data.root_quat_w  # [N, 4] (w, x, y, z)
    
    # Convert to rotation matrix and check z-axis alignment
    # For upright cube, z-axis should point up (0, 0, 1)
    # We can approximate this by checking if |quat.w| is close to 1 or 0
    # For small rotations from upright: quat ≈ (1, 0, 0, 0) or (0, 0, 0, 1)
    
    # Simpler check: penalize if quaternion deviates from upright
    # Upright quaternions: (±1, 0, 0, 0) or (0, ±1, 0, 0) or (0, 0, ±1, 0) or (0, 0, 0, ±1)
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    
    # Check deviation from identity quaternion (1, 0, 0, 0)
    deviation = torch.abs(w - 1.0) + torch.abs(x) + torch.abs(y) + torch.abs(z)
    
    return -0.1 * deviation


def energy_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalty for high joint velocities (energy consumption).
    
    Encourages energy-efficient movements.
    """
    robot: Articulation = env.scene["robot"]
    
    # Sum of squared joint velocities
    joint_vel = robot.data.joint_vel
    energy = torch.sum(torch.square(joint_vel), dim=1)
    
    return -0.001 * energy


# ============================================================================
# Composite Rewards (for easier config)
# ============================================================================

def stack_2cubes_shaped_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
) -> torch.Tensor:
    """Complete shaped reward for 2-cube stacking task.
    
    Combines multiple reward terms with appropriate weights.
    """
    # Stage 1: Reach cube_2
    reach_reward = 1.0 * gripper_distance_to_cube(env, ee_frame_cfg, cube_2_cfg)
    
    # Stage 2: Grasp cube_2
    grasp_reward = 2.0 * grasp_bonus(env, robot_cfg, ee_frame_cfg, cube_2_cfg)
    
    # Stage 3: Lift cube_2
    lift_reward = 2.0 * cube_lift_reward(env, cube_2_cfg, target_height=0.15)
    
    # Stage 4: Stack cube_2 on cube_1
    stack_reward = 10.0 * cube_stacking_reward(env, cube_2_cfg, cube_1_cfg)
    
    # Stage 5: Success
    success_reward = success_bonus_2cubes(env, robot_cfg, cube_1_cfg, cube_2_cfg)
    
    # Penalties
    action_penalty = action_rate_penalty(env)
    
    return reach_reward + grasp_reward + lift_reward + stack_reward + success_reward + action_penalty