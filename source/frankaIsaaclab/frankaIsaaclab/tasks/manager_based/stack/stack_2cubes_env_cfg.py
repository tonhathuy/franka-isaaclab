# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp
from .mdp import franka_stack_events
from .stack_env_cfg import StackEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from frankaIsaaclab.robots.franka import FRANKA_PANDA_CFG  # isort: skip


@configclass
class EventCfg2Cubes:
    """Configuration for events - 2 cubes version."""

    init_franka_arm_pose = EventTerm(
        func=franka_stack_events.set_default_joint_pose,
        mode="reset",
        params={
            "default_pose": [0.0444, -0.1894, -0.1107, -2.5148, 0.0044, 2.3775, 0.6952, 0.0400, 0.0400],
        },
    )

    randomize_franka_joint_state = EventTerm(
        func=franka_stack_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.02,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # Only 2 cubes now!
    randomize_cube_positions = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {"x": (0.4, 0.6), "y": (-0.10, 0.10), "z": (0.0203, 0.0203), "yaw": (-1.0, 1.0)},
            "min_separation": 0.1,
            "asset_cfgs": [SceneEntityCfg("cube_1"), SceneEntityCfg("cube_2")],  # Only 2 cubes
        },
    )


@configclass
class RewardsCfg2Cubes:
    """Reward terms for 2 cubes stacking task."""
    
    # Stage 1: Reach cube_2
    gripper_to_cube2 = RewTerm(
        func=mdp.gripper_distance_to_cube,
        weight=1.0,
        params={"cube_cfg": SceneEntityCfg("cube_2")},
    )
    
    # Stage 2: Lift cube_2
    lift_cube2 = RewTerm(
        func=mdp.cube_lift_reward,
        weight=2.0,
        params={"cube_cfg": SceneEntityCfg("cube_2"), "target_height": 0.15},
    )
    
    # Stage 3: Stack cube_2 on cube_1
    stack_cube2_on_cube1 = RewTerm(
        func=mdp.cube_stacking_reward,
        weight=10.0,
        params={
            "upper_cube_cfg": SceneEntityCfg("cube_2"),
            "lower_cube_cfg": SceneEntityCfg("cube_1"),
        },
    )
    
    # Bonus for success
    success_bonus = RewTerm(
        func=mdp.success_bonus_2cubes,
        weight=1.0,
    )
    
    # Penalties
    action_rate_penalty = RewTerm(
        func=mdp.action_rate_penalty,
        weight=0.01,
    )


@configclass
class TerminationsCfg2Cubes:
    """Termination terms for 2 cubes task."""
        
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    cube_1_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube_1")},
    )
    
    cube_2_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube_2")},
    )
    
    success = DoneTerm(func=mdp.two_cubes_stacked)


# @configclass
# class ObservationsCfg2Cubes:
#     """Observation specifications for 2 cubes task."""
#     from isaaclab.managers import ObservationGroupCfg as ObsGroup
#     from isaaclab.managers import ObservationTermCfg as ObsTerm

#     @configclass
#     class PolicyCfg(ObsGroup):
#         """Observations for policy group."""

#         actions = ObsTerm(func=mdp.last_action)
#         joint_pos = ObsTerm(func=mdp.joint_pos_rel)
#         joint_vel = ObsTerm(func=mdp.joint_vel_rel)
#         # Simplified object observations for 2 cubes
#         cube_1_pos = ObsTerm(
#             func=mdp.object_position_in_robot_root_frame,
#             params={"object_cfg": SceneEntityCfg("cube_1")},
#         )
#         cube_2_pos = ObsTerm(
#             func=mdp.object_position_in_robot_root_frame,
#             params={"object_cfg": SceneEntityCfg("cube_2")},
#         )
#         eef_pos = ObsTerm(func=mdp.ee_frame_pos)
#         eef_quat = ObsTerm(func=mdp.ee_frame_quat)
#         gripper_pos = ObsTerm(func=mdp.gripper_pos)

#         def __post_init__(self):
#             self.enable_corruption = False
#             self.concatenate_terms = True

#     @configclass
#     class SubtaskCfg(ObsGroup):
#         """Observations for subtask tracking."""

#         cube_2_grasped = ObsTerm(
#             func=mdp.object_grasped,
#             params={
#                 "robot_cfg": SceneEntityCfg("robot"),
#                 "ee_frame_cfg": SceneEntityCfg("ee_frame"),
#                 "object_cfg": SceneEntityCfg("cube_2"),
#             },
#         )
        
#         cubes_stacked = ObsTerm(
#             func=mdp.object_stacked,
#             params={
#                 "robot_cfg": SceneEntityCfg("robot"),
#                 "upper_object_cfg": SceneEntityCfg("cube_2"),
#                 "lower_object_cfg": SceneEntityCfg("cube_1"),
#             },
#         )

#         def __post_init__(self):
#             self.enable_corruption = False
#             self.concatenate_terms = False

#     # observation groups
#     policy: PolicyCfg = PolicyCfg()
#     subtask_terms: SubtaskCfg = SubtaskCfg()

@configclass
class ObservationsCfg2Cubes:
    """Observation specifications for 2 cubes task."""
    
    @configclass
    class PolicyCfg(ObsGroup):  # Sử dụng ObsGroup đã import ở đầu file
        """Observations for policy group."""

        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        
        # Use existing object observation function
        object = ObsTerm(
            func=mdp.object_obs,
            params={
                "cube_1_cfg": SceneEntityCfg("cube_1"),
                "cube_2_cfg": SceneEntityCfg("cube_2"),
                "cube_3_cfg": SceneEntityCfg("cube_2"),  # Reuse cube_2 since we only have 2 cubes
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            }
        )
        
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class SubtaskCfg(ObsGroup):  # Sử dụng ObsGroup đã import ở đầu file
        """Observations for subtask tracking."""

        cube_2_grasped = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("cube_2"),
            },
        )
        
        cubes_stacked = ObsTerm(
            func=mdp.object_stacked,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "upper_object_cfg": SceneEntityCfg("cube_2"),
                "lower_object_cfg": SceneEntityCfg("cube_1"),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class FrankaStack2CubesEnvCfg(StackEnvCfg):
    """Configuration for Franka stacking 2 cubes task."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Override episode length - shorter for simpler task
        self.episode_length_s = 20.0  # 20 seconds instead of 30

        # Set events for 2 cubes
        self.events = EventCfg2Cubes()
        
        # Set rewards for 2 cubes
        self.rewards = RewardsCfg2Cubes()
        
        # Set terminations for 2 cubes
        self.terminations = TerminationsCfg2Cubes()
        
        # Set observations for 2 cubes (optional - can use parent's)
        self.observations = ObservationsCfg2Cubes()

        # Set Franka as robot
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]

        # Add semantics to table
        self.scene.table.spawn.semantic_tags = [("class", "table")]

        # Add semantics to ground
        self.scene.plane.semantic_tags = [("class", "ground")]

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )
        # utilities for gripper status check
        self.gripper_joint_names = ["panda_finger_.*"]
        self.gripper_open_val = 0.04
        self.gripper_threshold = 0.005

        # Rigid body properties for cubes
        cube_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        )

        # ONLY 2 CUBES!
        self.scene.cube_1 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_1",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, -0.1, 0.0203], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=cube_properties,
                semantic_tags=[("class", "cube_1")],
            ),
        )
        self.scene.cube_2 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_2",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0.1, 0.0203], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=cube_properties,
                semantic_tags=[("class", "cube_2")],
            ),
        )
        # NO CUBE_3!

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
                    name="tool_rightfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
                    name="tool_leftfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
            ],
        )