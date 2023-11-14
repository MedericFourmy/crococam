"""
Wrapper class for load a URDF model in both Pinocchio and Bullet.

Derived from https://github.com/MeMory-of-MOtion/sobec/blob/v1.4.0/python/sobec/pinbullet.py
-> removed Talos specific codes
-> partial support for both free-flyer and fixed base robots
"""

import pinocchio as pin
import pybullet as pyb
import pybullet_data
import numpy as np
import example_robot_data as robex
from robot_utils import freezed_robot


class SimuProxy:
    def __init__(self):
        self.readyForSimu = False

    def init(self, dt_sim: float, robot_name: str, fixed_joint_names: list[str] = None, visual: bool = True, disable_joint_limits: bool = False):
        self.loadRobotFromErd(robot_name)
        pyb_mode = pyb.GUI if visual else pyb.DIRECT
        self.loadBulletModel(dt_sim, pyb_mode, disable_joint_limits)
        if fixed_joint_names is not None:
            self.freeze(fixed_joint_names)
        
        self.setTorqueControlMode()

    def loadRobotFromErd(self, name):
        inst = robex.ROBOTS[name]()

        self.robot = inst.robot

        self.urdf_path = inst.df_path
        self.srdf_path = inst.srdf_path
        self.free_flyer = inst.free_flyer
        self.ref_posture = inst.ref_posture

        # TODO: Useful?
        # pin.loadRotorParameters(self.robot.model, self.srdf_path, False)
        # pin.loadReferenceConfigurations(self.robot.model, self.srdf_path, False)

        self.setPinocchioFinalizationTricks()

    def setPinocchioFinalizationTricks(self):
        # Add free flyers joint limits ... WHY?
        if self.free_flyer:
            self.robot.model.upperPositionLimit[:7] = 1
            self.robot.model.lowerPositionLimit[:7] = -1

        self.robot.model.armature = (
            self.robot.model.rotorInertia * self.robot.model.rotorGearRatio**2
        )
        self.robot.q0 = self.robot.model.referenceConfigurations[self.ref_posture]

        # state variables for external force torques 
        # TODO: use pybullet API instead
        self.tau_fext = np.zeros(self.robot.model.nv)

    ################################################################################
    ################################################################################
    # Load bullet model
    def loadBulletModel(self, dt_sim, guiOpt=pyb.DIRECT, disable_joint_limits=True):

        self.bulletClient = pyb.connect(guiOpt)

        pyb.setTimeStep(dt_sim)

        # Set gravity (disabled by default in Bullet)
        pyb.setGravity(*(self.robot.model.gravity.linear))

        # Load horizontal plane
        pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.planeId = pyb.loadURDF("plane.urdf")

        if self.free_flyer:
            pose_root = self.robot.q0[:7]
        else:
            pose_root = [0,0,0, 0,0,0,1]
            
        self.robotId = pyb.loadURDF(
            self.urdf_path,
            pose_root[:3],
            pose_root[3:7],
            useFixedBase=not self.free_flyer,
        )

        if self.free_flyer:
            # Magic translation from bullet where the basis center is shifted
            self.localInertiaPos = pyb.getDynamicsInfo(self.robotId, -1)[3]

        self.setBulletFinalizationTrics()

        if disable_joint_limits:
            for idx in self.bullet_names2indices.values():
                pyb.changeDynamics(
                    self.robotId,
                    idx,
                    jointLowerLimit=-1000,
                    jointUpperLimit=1000,
                )

        nq_minus_nqa = 7 if self.free_flyer else 0 
        for ipin, ibul in enumerate(self.bulletCtrlJointsInPinOrder):
            pyb.enableJointForceTorqueSensor(1, ipin, True)
            pyb.resetJointState(self.robotId, ibul, self.robot.q0[nq_minus_nqa + ipin])

    def setBulletFinalizationTrics(self):
        self.bullet_names2indices = {
            pyb.getJointInfo(1, i)[1].decode(): i for i in range(pyb.getNumJoints(self.robotId))
        }
        """
        For free-flyer robots:
        >>> print(list(r.model.names))
        ['universe',
        'root_joint',
        ...]

        For fixed base robots:
        >>> print(list(r.model.names))
        ['universe',
        ...]
        """
        actuated_joint_index_start = 1 + int(self.free_flyer)
        self.bulletCtrlJointsInPinOrder = np.array([
            self.bullet_names2indices[n] for n in self.robot.model.names[actuated_joint_index_start:]
        ])

    ################################################################################



    def freeze(self, fixed_joint_names):
        """
        TODO: not tested for free flyers
        """

        fixed_jids = [self.robot.model.getJointId(jname) for jname in fixed_joint_names]

        # Build a new reduced model
        self.rmodel_full = self.robot.model
        rmodel, [gmodel_col, gmodel_vis] = pin.buildReducedModel(
            self.rmodel_full, [self.robot.collision_model, self.robot.visual_model],
            fixed_jids, self.robot.q0,
        )
        fixed_jids_min_universe = np.array(fixed_jids) - 1
        q0_fixed = self.robot.q0[fixed_jids_min_universe]
        nq_minus_nqa = 7 if self.free_flyer else 0 
        fixed_jids_pyb = self.bulletCtrlJointsInPinOrder[nq_minus_nqa+fixed_jids_min_universe]

        self.robot = pin.RobotWrapper(rmodel, gmodel_col, gmodel_vis)

        # Activate position control on the fixed joints so that they stay put in pybullet
        pyb.setJointMotorControlArray(
            self.robotId,
            jointIndices=fixed_jids_pyb,
            controlMode=pyb.POSITION_CONTROL,
            targetPositions=q0_fixed,
        )

        self.setPinocchioFinalizationTricks()
        self.setBulletFinalizationTrics()

    def setTorqueControlMode(self):
        """
        Disable default position controller in torque controlled joints
        Default controller will take care of other joints
        """
        pyb.setJointMotorControlArray(
            self.robotId,
            jointIndices=self.bulletCtrlJointsInPinOrder,
            controlMode=pyb.VELOCITY_CONTROL,
            forces=[0.0 for _ in self.bulletCtrlJointsInPinOrder],
        )
        self.readyForSimu = True

    def changeFriction(self, names, lateralFriction=100, spinningFriction=30):
        for n in names:
            idx = self.bullet_names2indices[n]
            pyb.changeDynamics(
                self.robotId,
                idx,
                lateralFriction=lateralFriction,
                spinningFriction=spinningFriction,
            )

    def setGravity(self, g):
        self.robot.model.gravity.linear = g
        pyb.setGravity(*g)

    ################################################################################
    # Called in the sim loop
    ################################################################################

    def step(self, torques):
        assert self.readyForSimu

        pyb.setJointMotorControlArray(
            self.robotId,
            self.bulletCtrlJointsInPinOrder,
            controlMode=pyb.TORQUE_CONTROL,
            forces=torques + self.tau_fext,
        )
        pyb.stepSimulation()

        # reset external force automatically after each simulation step
        self.tau_fext = np.zeros(self.robot.model.nv)

    def getState(self):
        # Get articulated joint pos and vel
        xbullet = pyb.getJointStates(self.robotId, self.bulletCtrlJointsInPinOrder)
        q = [x[0] for x in xbullet]
        vq = [x[1] for x in xbullet]

        if self.free_flyer:
            # TODO: not much tested
            # Get basis pose
            p, quat = pyb.getBasePositionAndOrientation(self.robotId)
            # Get basis vel
            v, w = pyb.getBaseVelocity(self.robotId)

            # Concatenate into a single x vector
            x = np.concatenate([p, quat, q, v, w, vq])

            # Magic transformation of the basis translation, as classical in Bullet.
            x[:3] -= self.localInertiaPos

        else:
            x = np.concatenate([q, vq])
        return x

    def setState(self, x):
        """Set the robot to the desired states.
        Args:
            q (ndarray): Desired generalized positions.
            dq (ndarray): Desired generalized velocities.
        """
        q, v = x[:self.robot.model.nq], x[self.robot.model.nq:]

        nq_minus_nqa = 7 if self.free_flyer else 0 
        for ipin, ibul in enumerate(self.bulletCtrlJointsInPinOrder):
            pyb.resetJointState(self.robotId, ibul, q[nq_minus_nqa + ipin], v[nq_minus_nqa + ipin])

    def applyExternalForce(self, f, ee_frame, rf_frame=pin.LOCAL_WORLD_ALIGNED):
        # Store the torque due to exterior forces for simulation step

        x = self.getState()
        q = x[:self.robot.model.nq]
        self.robot.framesForwardKinematics(q)
        self.robot.computeJointJacobians(q)
        r = self.robot
        pin.updateFramePlacements(r.model, r.data)
        Jf = pin.getFrameJacobian(r.model, r.data, r.model.getFrameId(ee_frame), rf_frame)
        
        # The torques from different forces accumulate
        self.tau_fext += Jf.T @ f



def test_run_SimuProxy(SimuProxy, robot_name, fixed_joints):

    import time
    from example_robot_data import load

    dur_sim = 20.0

    noise_tau_scale = 0.0

    # force disturbance
    t1_fext, t2_fext = 2.0, 3.0
    # fext = np.zeros(6)
    fext = np.array([0, 30, 0, 0, 0, 0])
    frame_name = 'panda_link5'

    # Gains are tuned at max before instability for each dt and controller type
    dt_sim = 1./240
    # JSID OK
    Kp = 100
    Kd = 10


    N_sim = int(dur_sim/dt_sim)

    robot = load('panda')
    robot = freezed_robot(robot, fixed_joints)
    v0 = np.zeros(robot.model.nv)
    
    sim = SimuProxy()
    sim.init(dt_sim, robot_name, fixed_joints, visual=True)
    
    q_init = robot.q0.copy() + 0.4  # little offset from stable position
    v_init = v0.copy() + 0.5
    x_init = np.concatenate([q_init, v_init]) 
    sim.setState(x_init)

    print('\n==========================')
    print('Begin simulation + control')
    print(f'Length: {dur_sim} seconds')
    print(f'dt_sim: {1000*dt_sim} milliseconds')
    print('Apply force between ', t1_fext, t2_fext, ' seconds')
    print('   -> ', t1_fext/dt_sim, t2_fext/dt_sim, ' iterations')
    print('fext: ', fext)

    for i in range(N_sim):
        ts = i*dt_sim
        t1 = time.time()
        x = sim.getState()
        q, v = x[:robot.model.nq], x[robot.model.nq:]

        #########################
        # Joint Space Inverse Dynamics
        ddqd = - Kp*(q - robot.q0) - Kd*(v - v0)
        tau = pin.rnea(robot.model, robot.data, q, v, ddqd)

        if t1_fext < ts < t2_fext:
            sim.applyExternalForce(
                fext, frame_name, rf_frame=pin.LOCAL_WORLD_ALIGNED)

        sim.step(tau)

        delay = time.time() - t1
        if delay < dt_sim:
            time.sleep(dt_sim - delay)


if __name__ == "__main__":
    dt_sim = 1e-3
    robot_name = 'panda'
    fixed_indexes = []
    # fixed_indexes = [2,5,6,7]
    fixed_joints =  [f'panda_joint{i}' for i in fixed_indexes] + ['panda_finger_joint1', 'panda_finger_joint2']
    # fixed_joints = None

    np.set_printoptions(linewidth=150)
    test_run_SimuProxy(SimuProxy, robot_name, fixed_joints)