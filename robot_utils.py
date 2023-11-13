from example_robot_data import load
import pinocchio as pin
import numpy as np



def create_panda(fixed_indexes: list[int]=[]):
    """
    Return reduced panda robot with only controllable joints.

    fixed_indexes: optional indices of arm articulation to be fixed.
    """

    robot = load('panda')
    fixed_joints =  [f'panda_joint{i}' for i in fixed_indexes] + ['panda_finger_joint1', 'panda_finger_joint2']
    return freezed_robot(robot, fixed_joints)


def freezed_robot(robot: pin.RobotWrapper, fixed_joints: list[str], q0_full: np.ndarray=None):
    """
    Return a robot wrapper with reduced set of joints with respect to full model.

    
    """
    if q0_full is None:
        q0_full = robot.q0

    # Remove some joints from pinocchio model
    fixed_jids = [robot.model.getJointId(jname) for jname in fixed_joints] \
                if fixed_joints is not None else []
    
    # Ugly code to resize model and q0
    rmodel, [gmodel_col, gmodel_vis] = pin.buildReducedModel(
            robot.model, [robot.collision_model, robot.visual_model],
            fixed_jids, robot.q0,
        )

    robot = pin.RobotWrapper(rmodel, gmodel_col, gmodel_vis)
    fixed_jids_min_universe = np.array(fixed_jids) - 1
    selected_jids = np.array([i for i in range(len(q0_full)) if i not in fixed_jids_min_universe])
    robot.q0 = q0_full[selected_jids]

    return robot
