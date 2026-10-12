"""Isaac Lab articulation configuration for the reduced GR1T2 humanoid.

The referenced URDF contains 25 actuated joints: 10 arm, 12 leg, and 3
waist joints.  Head joints and wrist pitch/roll joints were removed from the
URDF before this configuration was created.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg,ImplicitActuatorCfg,DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg



# GR1T2_CFG = ArticulationCfg(
#     spawn=sim_utils.UrdfFileCfg(
#         asset_path=f"/home/wx/WX/Qmini/Asset/gr1t2_description/urdf/GR1T2_new.urdf",
#         fix_base=False,
#         activate_contact_sensors=True,
#         replace_cylinders_with_capsules=True,
#         joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
#             gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
#         ),
#         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
#             enabled_self_collisions=True,
#             solver_position_iteration_count=8,
#             solver_velocity_iteration_count=4,
#         ),
#         rigid_props=sim_utils.RigidBodyPropertiesCfg(
#             disable_gravity=False,
#             retain_accelerations=False,
#             linear_damping=0.0,
#             angular_damping=0.0,
#             max_linear_velocity=1000.0,
#             max_angular_velocity=1000.0,
#             max_depenetration_velocity=1.0,
#         ),
#     ),
#     init_state=ArticulationCfg.InitialStateCfg(
#         # Approximate standing height for the headless GR1T2 model.
#         pos=(0.0, 0.0, 1.0),
#         joint_pos={
#             # 左腿
#             "l_hip_roll": 0.0,
#             "l_hip_yaw": 0.0,
#             "l_hip_pitch": -0.15,
#             "l_knee_pitch": 0.30,
#             "l_ankle_pitch": -0.15,
#             "l_ankle_roll": 0.0,
#             # 右腿
#             "r_hip_roll": 0.0,
#             "r_hip_yaw": 0.0,
#             "r_hip_pitch": -0.15,
#             "r_knee_pitch": 0.30,
#             "r_ankle_pitch": -0.15,
#             "r_ankle_roll": 0.0,
#             # 腰部
#             "joint_waist_yaw": 0.0,
#             "joint_waist_pitch": 0.0,
#             "joint_waist_roll": 0.0,
#             # 左臂
#             "l_shoulder_pitch": 0.0,
#             "l_shoulder_roll": 0.0,
#             "l_shoulder_yaw": 0.0,
#             "l_elbow_pitch": 0.0,
#             "l_wrist_yaw": 0.0,
#             # 右臂
#             "r_shoulder_pitch": 0.0,
#             "r_shoulder_roll": 0.0,
#             "r_shoulder_yaw": 0.0,
#             "r_elbow_pitch": 0.0,
#             "r_wrist_yaw": 0.0,
#         },
#         joint_vel={".*": 0.0},
#     ),
#     soft_joint_pos_limit_factor=0.9,
#     actuators={
#         "left_arm": ImplicitActuatorCfg(
#             joint_names_expr=[
#                 "l_shoulder_pitch",
#                 "l_shoulder_roll",
#                 "l_shoulder_yaw",
#                 "l_elbow_pitch",
#                 "l_wrist_yaw",
#             ],
#             effort_limit=900.0,
#             velocity_limit=25.0,
#             stiffness=200.0,
#             damping=10.0,
#             armature=0.01,
#             friction=0.0,
#         ),
#         "right_arm": ImplicitActuatorCfg(
#             joint_names_expr=[
#                 "r_shoulder_pitch",
#                 "r_shoulder_roll",
#                 "r_shoulder_yaw",
#                 "r_elbow_pitch",
#                 "r_wrist_yaw",
#             ],
#             effort_limit=900.0,
#             velocity_limit=25.0,
#             stiffness=200.0,
#             damping=10.0,
#             armature=0.01,
#             friction=0.0,
#         ),
#         "left_leg": ImplicitActuatorCfg(
#             joint_names_expr=[
#                 "l_hip_roll",
#                 "l_hip_yaw",
#                 "l_hip_pitch",
#                 "l_knee_pitch",
#                 "l_ankle_pitch",
#                 "l_ankle_roll",
#             ],
#             effort_limit=900.0,
#             velocity_limit=25.0,
#             stiffness=200.0,
#             damping=10.0,
#             armature=0.01,
#             friction=0.0,
#         ),
#         "right_leg": ImplicitActuatorCfg(
#             joint_names_expr=[
#                 "r_hip_roll",
#                 "r_hip_yaw",
#                 "r_hip_pitch",
#                 "r_knee_pitch",
#                 "r_ankle_pitch",
#                 "r_ankle_roll",
#             ],
#             effort_limit=900.0,
#             velocity_limit=25.0,
#             stiffness=200.0,
#             damping=10.0,
#             armature=0.01,
#             friction=0.0,
#         ),
#         "waist": ImplicitActuatorCfg(
#             joint_names_expr=[
#                 "joint_waist_yaw",
#                 "joint_waist_pitch",
#                 "joint_waist_roll",
#             ],
#             effort_limit=900.0,
#             velocity_limit=25.0,
#             stiffness=200.0,
#             damping=10.0,
#             armature=0.01,
#             friction=0.0,
#         ),
#     },
# )




GR1T2_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=f"/home/wx/WX/Qmini/Asset/gr1t2_description/urdf/GR1T2_new.urdf",
        fix_base=False,
        activate_contact_sensors=True,
        replace_cylinders_with_capsules=True,
        joint_drive = sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
        articulation_props = sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.85),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators = {
            "left_leg": DCMotorCfg(
                joint_names_expr=[
                    "l_hip_roll",
                    "l_hip_yaw",
                    "l_hip_pitch",
                    "l_knee_pitch",
                    "l_ankle_pitch",
                    "l_ankle_roll",
                ],
                effort_limit={
                    "l_hip_roll": 48.0,
                    "l_hip_yaw": 66.0,
                    "l_hip_pitch": 225.0,
                    "l_knee_pitch": 225.0,
                    "l_ankle_pitch": 15.0,
                    "l_ankle_roll": 30.0,
                },
                velocity_limit={
                    "l_hip_roll": 12.15,
                    "l_hip_yaw": 16.76,
                    "l_hip_pitch": 37.38,
                    "l_knee_pitch": 37.38,
                    "l_ankle_pitch": 20.32,
                    "l_ankle_roll": 20.32,
                },
                stiffness={
                    "l_hip_roll": 100.0,
                    "l_hip_yaw": 60.0,
                    "l_hip_pitch": 150.0,
                    "l_knee_pitch": 160.0,
                    "l_ankle_pitch": 40.0,
                    "l_ankle_roll": 30.0,
                },
                damping={
                    "l_hip_roll": 5.0,
                    "l_hip_yaw": 3.0,
                    "l_hip_pitch": 6.0,
                    "l_knee_pitch": 6.0,
                    "l_ankle_pitch": 2.0,
                    "l_ankle_roll": 1.5,
                },
                armature=0.01,
                friction=0.0,
                saturation_effort=225.0,
            ),
            "right_leg": DCMotorCfg(
                joint_names_expr=[
                    "r_hip_roll",
                    "r_hip_yaw",
                    "r_hip_pitch",
                    "r_knee_pitch",
                    "r_ankle_pitch",
                    "r_ankle_roll",
                ],
                effort_limit={
                    "r_hip_roll": 48.0,
                    "r_hip_yaw": 66.0,
                    "r_hip_pitch": 225.0,
                    "r_knee_pitch": 225.0,
                    "r_ankle_pitch": 15.0,
                    "r_ankle_roll": 30.0,
                },
                velocity_limit={
                    "r_hip_roll": 12.15,
                    "r_hip_yaw": 16.76,
                    "r_hip_pitch": 37.38,
                    "r_knee_pitch": 37.38,
                    "r_ankle_pitch": 20.32,
                    "r_ankle_roll": 20.32,
                },
                stiffness={
                    "r_hip_roll": 100.0,
                    "r_hip_yaw": 60.0,
                    "r_hip_pitch": 150.0,
                    "r_knee_pitch": 160.0,
                    "r_ankle_pitch": 40.0,
                    "r_ankle_roll": 30.0,
                },
                damping={
                    "r_hip_roll": 5.0,
                    "r_hip_yaw": 3.0,
                    "r_hip_pitch": 6.0,
                    "r_knee_pitch": 6.0,
                    "r_ankle_pitch": 2.0,
                    "r_ankle_roll": 1.5,
                },
                armature=0.01,
                friction=0.0,
                saturation_effort=225.0,
            ),
            "waist": DCMotorCfg(
                joint_names_expr=[
                    "joint_waist_yaw",
                    "joint_waist_pitch",
                    "joint_waist_roll",
                ],
                effort_limit=66.0,
                velocity_limit=16.76,
                stiffness={
                    "joint_waist_yaw": 60.0,
                    "joint_waist_pitch": 120.0,
                    "joint_waist_roll": 100.0,
                },
                damping={
                    "joint_waist_yaw": 3.0,
                    "joint_waist_pitch": 6.0,
                    "joint_waist_roll": 5.0,
                },
                armature=0.01,
                friction=0.0,
                saturation_effort=66.0,
            ),
            "left_arm": DCMotorCfg(
                joint_names_expr=[
                    "l_shoulder_pitch",
                    "l_shoulder_roll",
                    "l_shoulder_yaw",
                    "l_elbow_pitch",
                    "l_wrist_yaw",
                ],
                effort_limit={
                    "l_shoulder_pitch": 38.0,
                    "l_shoulder_roll": 38.0,
                    "l_shoulder_yaw": 30.0,
                    "l_elbow_pitch": 30.0,
                    "l_wrist_yaw": 10.2,
                },
                velocity_limit={
                    "l_shoulder_pitch": 9.11,
                    "l_shoulder_roll": 9.11,
                    "l_shoulder_yaw": 7.33,
                    "l_elbow_pitch": 7.33,
                    "l_wrist_yaw": 24.4,
                },
                stiffness={
                    "l_shoulder_pitch": 60.0,
                    "l_shoulder_roll": 50.0,
                    "l_shoulder_yaw": 40.0,
                    "l_elbow_pitch": 40.0,
                    "l_wrist_yaw": 15.0,
                },
                damping={
                    "l_shoulder_pitch": 3.0,
                    "l_shoulder_roll": 2.5,
                    "l_shoulder_yaw": 2.0,
                    "l_elbow_pitch": 2.0,
                    "l_wrist_yaw": 0.8,
                },
                armature=0.01,
                friction=0.0,
                saturation_effort=38.0,
            ),
            "right_arm": DCMotorCfg(
                joint_names_expr=[
                    "r_shoulder_pitch",
                    "r_shoulder_roll",
                    "r_shoulder_yaw",
                    "r_elbow_pitch",
                    "r_wrist_yaw",
                ],
                effort_limit={
                    "r_shoulder_pitch": 38.0,
                    "r_shoulder_roll": 38.0,
                    "r_shoulder_yaw": 30.0,
                    "r_elbow_pitch": 30.0,
                    "r_wrist_yaw": 10.2,
                },
                velocity_limit={
                    "r_shoulder_pitch": 9.11,
                    "r_shoulder_roll": 9.11,
                    "r_shoulder_yaw": 7.33,
                    "r_elbow_pitch": 7.33,
                    "r_wrist_yaw": 24.4,
                },
                stiffness={
                    "r_shoulder_pitch": 60.0,
                    "r_shoulder_roll": 50.0,
                    "r_shoulder_yaw": 40.0,
                    "r_elbow_pitch": 40.0,
                    "r_wrist_yaw": 15.0,
                },
                damping={
                    "r_shoulder_pitch": 3.0,
                    "r_shoulder_roll": 2.5,
                    "r_shoulder_yaw": 2.0,
                    "r_elbow_pitch": 2.0,
                    "r_wrist_yaw": 0.8,
                },
                armature=0.01,
                friction=0.0,
                saturation_effort=38.0,
            ),
        }
)

# Backward-compatible alias for tasks that still import the old symbol.

JOINT_NAMES = [
    # 左腿 6
    "l_hip_roll",
    "l_hip_yaw",
    "l_hip_pitch",
    "l_knee_pitch",
    "l_ankle_pitch",
    "l_ankle_roll",

    # 右腿 6
    "r_hip_roll",
    "r_hip_yaw",
    "r_hip_pitch",
    "r_knee_pitch",
    "r_ankle_pitch",
    "r_ankle_roll",

    # 腰部 3
    "joint_waist_yaw",
    "joint_waist_pitch",
    "joint_waist_roll",

    # 左臂 5
    "l_shoulder_pitch",
    "l_shoulder_roll",
    "l_shoulder_yaw",
    "l_elbow_pitch",
    "l_wrist_yaw",

    # 右臂 5
    "r_shoulder_pitch",
    "r_shoulder_roll",
    "r_shoulder_yaw",
    "r_elbow_pitch",
    "r_wrist_yaw",
]

FIXED_JOINT_NAMES = [
    "joint_torso",
    "joint_imu",
]

RIGID_BODY_NAMES = [
    # 根节点
    "base",

    # 左腿
    "l_thigh_roll",
    "l_thigh_yaw",
    "l_thigh_pitch",
    "l_shank_pitch",
    "l_foot_pitch",
    "l_foot_roll",

    # 右腿
    "r_thigh_roll",
    "r_thigh_yaw",
    "r_thigh_pitch",
    "r_shank_pitch",
    "r_foot_pitch",
    "r_foot_roll",

    # 腰部 / 躯干
    "link_waist_yaw",
    "link_waist_pitch",
    "link_waist_roll",
    "link_torso",

    # 左臂
    "l_upper_arm_pitch",
    "l_upper_arm_roll",
    "l_upper_arm_yaw",
    "l_lower_arm_pitch",
    "l_hand_yaw",

    # 右臂
    "r_upper_arm_pitch",
    "r_upper_arm_roll",
    "r_upper_arm_yaw",
    "r_lower_arm_pitch",
    "r_hand_yaw",

    # IMU
    "link_imu",
]



############### ---------函数用变量----------- #############################

LEG_JOINT_NAMES = [
    # Left leg
    "l_hip_roll",
    "l_hip_yaw",
    "l_hip_pitch",
    "l_knee_pitch",
    "l_ankle_pitch",
    "l_ankle_roll",

    # Right leg
    "r_hip_roll",
    "r_hip_yaw",
    "r_hip_pitch",
    "r_knee_pitch",
    "r_ankle_pitch",
    "r_ankle_roll",
]


WAIST_JOINT_NAMES = [
    "joint_waist_yaw",
    "joint_waist_pitch",
    "joint_waist_roll",
]


ARM_JOINT_NAMES = [
    # Left arm
    "l_shoulder_pitch",
    "l_shoulder_roll",
    "l_shoulder_yaw",
    "l_elbow_pitch",
    "l_wrist_yaw",

    # Right arm
    "r_shoulder_pitch",
    "r_shoulder_roll",
    "r_shoulder_yaw",
    "r_elbow_pitch",
    "r_wrist_yaw",
]
