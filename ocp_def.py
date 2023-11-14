


from dataclasses import dataclass
import numpy as np
import crocoddyl as croc
import pinocchio as pin

@dataclass
class ConfigOCP:
    ##############
    # Task weigths
    ##############
    w_frame_terminal = 100.0
    w_frame_running = 1

    w_frame_vel_running = 1.0
    w_frame_vel_terminal = 1.0
    diag_ee_vel = np.ones(6)

    # State regularization
    w_x_reg_running = 0.1
    w_x_reg_terminal = 1
    scale_q_vs_v_reg = 0.1

    # Control regularization
    w_u_reg = 0.01
    diag_u_reg = np.ones(7)
    armature_scale = 0.0  # creates instabilities

    # Joint limits
    w_joint_limits_running = 0.0
    w_joint_limits_terminal = 0.0

    ee_name = 'panda_hand'

    # Number of shooting nodes
    T = 100
    dt = 1e-2

    verbose=False


class OCP:

    def __init__(self, model: pin.Model, cfg: ConfigOCP):
        # # # # # # # # # # # # # # #
        ###  SETUP croc OCP  ###
        # # # # # # # # # # # # # # #

        self.model = model.copy()
        x0_dummy = np.zeros(self.model.nq+self.model.nv)
        self.cfg = cfg

        ee_frame_id = model.getFrameId(cfg.ee_name)

        # State and actuation model
        state = croc.StateMultibody(self.model)
        actuation = croc.ActuationModelFull(state)

        # Bounds
        self.x_limits_lower = np.concatenate([model.lowerPositionLimit, -model.velocityLimit])
        self.x_limits_upper = np.concatenate([model.upperPositionLimit,  model.velocityLimit])
        print('self.x_limits_lower:',self.x_limits_lower)
        print('self.x_limits_upper:',self.x_limits_upper)
        bounds = croc.ActivationBounds(self.x_limits_lower, self.x_limits_upper, beta=1)

        ################################################
        # Cost terms common between running and terminal 

        # end translation cost: r(x_i, u_i) = translation(q_i) - t_ref
        oMe_dummy = pin.SE3.Identity()
        frameGoalResidual = croc.ResidualModelFramePlacement(state, ee_frame_id, oMe_dummy)
        frameGoalCost = croc.CostModelResidual(state, frameGoalResidual)

        frameVelCost = croc.CostModelResidual(state, 
                                              croc.ActivationModelWeightedQuad(cfg.diag_ee_vel**2), 
                                              croc.ResidualModelFrameVelocity(state, ee_frame_id, pin.Motion.Zero(), pin.LOCAL_WORLD_ALIGNED, actuation.nu))

        # Control regularization cost: r(x_i, u_i) = tau_i - g(q_i)
        uRegCost = croc.CostModelResidual(state, 
                                            croc.ActivationModelWeightedQuad(cfg.diag_u_reg**2), 
                                            croc.ResidualModelControlGrav(state, actuation.nu))

        # Joint limits cost
        jointLimitCost = croc.CostModelResidual(state, 
                                                croc.ActivationModelQuadraticBarrier(bounds), 
                                                croc.ResidualModelState(state, actuation.nu))
        ###################


        ###################
        # Running only
        # State regularization cost: r(x_i, u_i) = diff(x_i, x_ref)
        diag_x_reg_running = np.array(
            self.model.nq*[cfg.scale_q_vs_v_reg] + self.model.nv*[1.0]
        )
        xRegCost = croc.CostModelResidual(state, 
                                            croc.ActivationModelWeightedQuad(diag_x_reg_running**2), 
                                            croc.ResidualModelState(state, x0_dummy, actuation.nu))
        ###################

        ###############
        # Running costs
        runningModel_lst = []
        for _ in range(cfg.T):
            runningCostModel = croc.CostModelSum(state)
            runningCostModel.addCost('stateReg', xRegCost, cfg.w_x_reg_running)
            runningCostModel.addCost('ctrlRegGrav', uRegCost, cfg.w_u_reg)
            runningCostModel.addCost('placement', frameGoalCost, cfg.w_frame_running)
            runningCostModel.addCost('jointLimit', jointLimitCost, cfg.w_joint_limits_running)
            runningCostModel.addCost('ee_vel', frameVelCost, cfg.w_frame_vel_running)
            
            # DAM: Continuous cost functions with continuous dynamics
            # IAE: Euler integration of continuous dynamics
            running_DAM = croc.DifferentialActionModelFreeFwdDynamics(state, actuation, runningCostModel)
            runningModel = croc.IntegratedActionModelEuler(running_DAM, cfg.dt)
            # Model actuator's inertia
            runningModel.differential.armature = cfg.armature_scale*np.ones(model.nv)
            runningModel_lst.append(runningModel)
        
        ###############
        # Terminal cost
        # !! weights scale has a different meaning here since
        # weights in the running cost are multiplied by dt 
        ###############
        
        diag_x_reg_terminal = np.array(
            self.model.nq*[cfg.scale_q_vs_v_reg] + self.model.nv*[1.0]
        )
        xRegCost = croc.CostModelResidual(state, 
                                          croc.ActivationModelWeightedQuad(diag_x_reg_terminal**2), 
                                          croc.ResidualModelState(state, x0_dummy, actuation.nu))
        
        terminalCostModel = croc.CostModelSum(state)
        terminalCostModel.addCost('stateReg', xRegCost,         cfg.dt*cfg.w_x_reg_terminal)
        terminalCostModel.addCost('placement', frameGoalCost,   cfg.dt*cfg.w_frame_terminal)
        terminalCostModel.addCost('jointLimit', jointLimitCost, cfg.dt*cfg.w_joint_limits_terminal)
        terminalCostModel.addCost('ee_vel', frameVelCost,       cfg.dt*cfg.w_frame_vel_terminal)

        terminal_DAM = croc.DifferentialActionModelFreeFwdDynamics(
            state, actuation, terminalCostModel
        )
        terminalModel = croc.IntegratedActionModelEuler(terminal_DAM, 0.0)
        # Optionally add armature to take into account actuator's inertia
        terminalModel.differential.armature = cfg.armature_scale*np.ones(model.nv)
        ###############

        # Create the shooting problem
        problem = croc.ShootingProblem(x0_dummy, runningModel_lst, terminalModel)

        # Create solver + callbacks
        self.ddp = croc.SolverFDDP(problem)
        if cfg.verbose:
            self.ddp.setCallbacks([croc.CallbackLogger(), croc.CallbackVerbose()])

    def quasistatic_init(self, x0):
        # Warm start : initial state + gravity compensation
        xs_init = (self.cfg.T + 1)*[x0]
        us_init = self.ddp.problem.quasiStatic(xs_init[:-1])
        return xs_init, us_init
    
    def set_initial_state(self, x0):
        self.ddp.problem.x0 = x0

    def set_ref(self, cost_name: str, ref):
        for i in range(self.cfg.T):
            running_costs_i = self.ddp.problem.runningModels[i].differential.costs
            running_costs_i.costs[cost_name].cost.residual.reference = ref
            running_costs_i.changeCostStatus(cost_name, True)

        terminal_costs = self.ddp.problem.terminalModel.differential.costs
        terminal_costs.costs[cost_name].cost.residual.reference = ref
        terminal_costs.changeCostStatus(cost_name, True)

    def set_ee_placement_ref(self, oMe: pin.SE3):
        assert isinstance(oMe, pin.SE3)
        self.set_ref('placement', oMe)

    def set_state_reg_ref(self, x0: np.ndarray):
        assert isinstance(x0, np.ndarray)
        assert len(x0) == self.model.nq + self.model.nv 
        self.set_ref('stateReg', x0)
