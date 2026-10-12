import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg,ImplicitActuatorCfg,DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

UMV_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path="/home/wx/WX/Qmini/Asset/UMV/umv.urdf",
        fix_base=False,
        activate_contact_sensors=True,
        replace_cylinders_with_capsules=False,
        force_usd_conversion=True,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
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
        pos=(0.0, 0.0, 0.35),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "q_w": 0.0,           # 动力后轮（连续旋转）
            "q_frontwheel": 0.0,  # 从动前轮（被动自由滚转）
            "q_s": 0.0,           # 转向偏角
            "q_n": 0.0,           # 颈部
            "q_h": 0.0,           # 头部
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        # 1. 动力后轮：连续旋转关节，设为速度/力矩驱动（stiffness=0，纯速度控制）
        "drive_wheel": ImplicitActuatorCfg(
            joint_names_expr=["q_w"],
            effort_limit=50.0,      # 后驱电机峰值力矩 (N·m)
            velocity_limit_sim=100.0,  # 仿真求解器最大转速 (rad/s)
            stiffness=0.0,          # 连续旋转关节必须为 0（无位置刚度回弹）
            damping=1.0,            # 电机速度阻尼
        ),
        # 2. 从动前轮：无驱动动力，纯自由被动滚转
        "passive_wheel": ImplicitActuatorCfg(
            joint_names_expr=["q_frontwheel"],
            effort_limit=0.0,       # 力矩上限设为 0，彻底切断主动输出
            velocity_limit_sim=100.0,
            stiffness=0.0,          # 无回弹刚度
            damping=0.01,           # 模拟轴承机械摩擦阻尼，可根据需要微调
        ),
        # 3. 转向机构（带有限位与定位刚度）
        "steering": ImplicitActuatorCfg(
            joint_names_expr=["q_s"],
            effort_limit=6.25,
            velocity_limit_sim=160.0,
            stiffness=60.0,
            damping=3.0,
        ),
        # 4. 上半身关节
        "upper_body": ImplicitActuatorCfg(
            joint_names_expr=["q_n", "q_h"],
            effort_limit=200.0,
            velocity_limit_sim=30.0,
            stiffness=150.0,
            damping=10.0,
        ),
    },
)
