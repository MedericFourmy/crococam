


import numpy as np
import crocoddyl as croc


def linear_interpolation(x, x1, x2, y1, y2):
    return y1 + ((x - x1) / (x2 - x1)) * (y2 - y1)

def tanh_interpolation(x, low, high, scale, shift=0):
    x_norm = linear_interpolation(x, 0, len(x), scale*(-1 - shift), scale*(1 - shift))
    return low + 0.5*high*(np.tanh(x_norm)+1)


class OCP:


    def __init__(self, model, x0, ee_name, oMe_goal, T, dt, goal_is_se3=True, verbose=False):
        # # # # # # # # # # # # # # #
        ###  SETUP croc OCP  ###
        # # # # # # # # # # # # # # #

        """
        Objects we need to define

        CostModelResidual defines a cost model as cost = a(r(x,u)) where
        r is a residual function, a is an activation model

        The class (as other Residual types) implements:
        - calc: computes the residual function
        - calcDiff: computes the residual derivatives
        Results are stored in a ResidualDataAbstract

        Default activation function is quadratic
        """
        model = model.copy()

        ee_frame_id = model.getFrameId(ee_name)

        # State and actuation model
        state = croc.StateMultibody(model)
        actuation = croc.ActuationModelFull(state)

        ###################
        # Create cost terms

        # end translation cost: r(x_i, u_i) = translation(q_i) - t_ref
        frameGoalResidual = None
        if goal_is_se3:
            frameGoalResidual = croc.ResidualModelFramePlacement(state, ee_frame_id, oMe_goal)
        else:
            frameGoalResidual = croc.ResidualModelFrameTranslation(state, ee_frame_id, oMe_goal.translation)
        frameGoalCost = croc.CostModelResidual(state, frameGoalResidual)


        ##############
        # Task weigths
        ##############
        # EE pose
        # w_running_frame_low = 100
        # w_running_frame_high = 100
        w_frame_terminal = 100.0
        w_running_frame_low = 0
        w_running_frame_high = 0
        # w_frame_terminal = 0
        
        # State regularization
        w_x_reg_running = 0.1
        w_x_reg_terminal = 1
        scale_q_vs_v_reg = 0.1

        # Control regularization
        w_u_reg_running = 0.01
        diag_u_reg_running = np.ones(model.nv)


        # State regularization
        diag_x_reg_running = np.array(
            model.nq*[scale_q_vs_v_reg] + model.nv*[1.0]
        )
        diag_x_reg_terminal = np.array(
            model.nq*[scale_q_vs_v_reg] + model.nv*[1.0]
        )

        # w_frame_schedule = linear_interpolation(np.arange(T), 0, T-1, w_running_frame_low, w_running_frame_high)
        w_frame_schedule = tanh_interpolation(np.arange(T), w_running_frame_low, w_running_frame_high, scale=5, shift=0.0)
        # w_frame_schedule = tanh_interpolation(np.arange(T), w_running_frame_low, w_running_frame_high, scale=8, shift=0.0)

        ###############
        # Running costs
        goal_cost_name = 'placement' if goal_is_se3 else 'translation'
        runningModel_lst = []
        for i in range(T):
            runningCostModel = croc.CostModelSum(state)

            # State regularization cost: r(x_i, u_i) = diff(x_i, x_ref)
            xRegCost = croc.CostModelResidual(state, 
                                                croc.ActivationModelWeightedQuad(diag_x_reg_running**2), 
                                                croc.ResidualModelState(state, x0, actuation.nu))

            # Control regularization cost: r(x_i, u_i) = tau_i - g(q_i)
            uRegCost = croc.CostModelResidual(state, 
                                                croc.ActivationModelWeightedQuad(diag_u_reg_running**2), 
                                                croc.ResidualModelControlGrav(state, actuation.nu))


            runningCostModel.addCost('stateReg', xRegCost, w_x_reg_running)
            runningCostModel.addCost('ctrlRegGrav', uRegCost, w_u_reg_running)
            runningCostModel.addCost(goal_cost_name, frameGoalCost, w_frame_schedule[i])
            # Create Differential Action Model (DAM), i.e. continuous dynamics and cost functions
            running_DAM = croc.DifferentialActionModelFreeFwdDynamics(
                state, actuation, runningCostModel
            )
            # Create Integrated Action Model (IAM), i.e. Euler integration of continuous dynamics and cost
            runningModel = croc.IntegratedActionModelEuler(running_DAM, dt)
            # Optionally add armature to take into account actuator's inertia
            # runningModel.differential.armature = 0.1*np.ones(model.nv)

            runningModel_lst.append(runningModel)
        
        ###############
        # Terminal cost
        # !! weights scale has a different meaning here since
        # weights in the running cost are multiplied by dt 
        ###############
        terminalCostModel = croc.CostModelSum(state)
        xRegCost = croc.CostModelResidual(state, 
                                            croc.ActivationModelWeightedQuad(diag_x_reg_terminal**2), 
                                            croc.ResidualModelState(state, x0, actuation.nu))
        # Control regularization cost: nu(x_i) = v_ee(x_i) - v_ee*
        # frameVelCost = croc.CostModelResidual(state, 
        #                                         croc.ActivationModelWeightedQuad(diag_vel_terminal**2), 
        #                                         croc.ResidualModelFrameVelocity(state, ee_frame_id, pin.Motion.Zero(), pin.LOCAL_WORLD_ALIGNED, actuation.nu))

        terminalCostModel.addCost('stateReg', xRegCost, w_x_reg_terminal)
        terminalCostModel.addCost(goal_cost_name, frameGoalCost, w_frame_terminal)
        # terminalCostModel.addCost('terminal_vel', frameVelCost, w_frame_vel_terminal)


        terminal_DAM = croc.DifferentialActionModelFreeFwdDynamics(
            state, actuation, terminalCostModel
        )
        terminalModel = croc.IntegratedActionModelEuler(terminal_DAM, 0.0)
        # Optionally add armature to take into account actuator's inertia
        # terminalModel.differential.armature = 0.1*np.ones(model.nv)

        # Create the shooting problem
        problem = croc.ShootingProblem(x0, runningModel_lst, terminalModel)

        # Create solver + callbacks
        self.ddp = croc.SolverFDDP(problem)
        if verbose:
            self.ddp.setCallbacks([croc.CallbackLogger(), croc.CallbackVerbose()])
