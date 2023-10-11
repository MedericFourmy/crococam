"""
Example script : Crocoddyl OCP with Panda arm 
static target reaching task

Directly inspired by https://github.com/skleff1994/minimal_examples_crocoddyl/blob/master/ocp_kuka_reaching.py

# Questions:
- How to update reference translation without rebuilding another ddp problem?
-> ddp.problem.terminalModel.differential.costs.costs['placement'].cost.residual.reference.translation = new_ref?
"""

from pathlib import Path
import numpy as np
np.set_printoptions(precision=4, linewidth=180)
import pinocchio as pin
import ocp_utils
from example_robot_data import load

from ocp_reaching import OCP
from pinbullet import freezed_robot

GOAL_IS_SE3 = True
VERBOSE = True
SAVE = False
GVIEWER = True
PLOT = True

SAVE_DIR = Path('tmp')
SAVE_DIR.mkdir(exist_ok=True)
LABEL = 'slow'


robot = load('panda')
ee_name = 'panda_link8'
fixed_indexes = []
# fixed_indexes = [3]
fixed_joints =  [f'panda_joint{i}' for i in fixed_indexes] + ['panda_finger_joint1', 'panda_finger_joint2']
# fixed_joints = None
robot = freezed_robot(robot, fixed_joints)


delta_trans = np.array([-0.31, -0.3, -0.0])
# delta_trans = np.array([-0.0, -0.0, -0.0])

# Number of shooting nodes
T = 25
dt_ocp = 1e-2  # seconds

oMe_0 = robot.framePlacement(robot.q0, robot.model.getFrameId(ee_name), update_kinematics=True)
oMe_goal = oMe_0.copy()
oMe_goal.translation += delta_trans
oMe_goal.rotation = np.eye(3)
print(oMe_0)

x0 = np.concatenate([robot.q0, np.zeros(robot.nv)])
ocp = OCP(robot.model, x0, ee_name, oMe_goal, T, dt_ocp, goal_is_se3=GOAL_IS_SE3, verbose=VERBOSE)


# Warm start : initial state + gravity compensation
xs_init = [x0 for i in range(T + 1)]
# TODO: check same as 
us_init = ocp.ddp.problem.quasiStatic(xs_init[:-1])
us_init_bis = np.array(
    [robot.gravity(robot.q0)]
)
ocp.ddp.solve(xs_init, us_init, maxiter=100, is_feasible=False)

# Extract DDP data and plot
ddp_data = ocp_utils.extract_ocp_data(ocp.ddp, ee_name)

#  Warm start using exactly the previous solution
ocp.ddp.problem.x0 = x0
xs_init = ocp.ddp.xs
us_init = ocp.ddp.us 
ocp.ddp.solve(xs_init, us_init, maxiter=100, is_feasible=False)
ddp_data = ocp_utils.extract_ocp_data(ocp.ddp, ee_name)

# solution joint trajectory
xs = np.array(ocp.ddp.xs)
q_final = xs[-1, : robot.model.nq]
oMe_fin = robot.framePlacement(
    q_final, robot.model.getFrameId(ee_name), update_kinematics=True
)

if SAVE:
    import os
    import pandas as pd

    def linear_interp(t_arr2, t_arr1, arr1):
        """
        Linear interpolation of numpy array.

        Each column of arr1 represents a time series, the row correspond to observations
        at successive time, t_arr1 is the original time array, t_arr2 is the new one.
        """
        arr2 = np.zeros((t_arr2.shape[0], arr1.shape[1]))
        for j in range(arr1.shape[1]):
            arr2[:,j] = np.interp(t_arr2, t_arr1, arr1[:,j])
        return arr2



    x_arr = np.array(ocp.ddp.xs)
    # one less control var in the trajectory than states
    q_arr = x_arr[:-1,:7]
    v_arr = x_arr[:-1,-7:]
    tau_arr = np.array(ocp.ddp.us)

    t_arr1 = dt_ocp*np.arange(T)
    dt_control = 1e-3
    t_arr2 = dt_control*np.arange(int(T*(dt_ocp/dt_control)))

    print(t_arr2.shape, t_arr1.shape, q_arr.shape)
    q_arr_ctrl =   linear_interp(t_arr2, t_arr1, q_arr)
    v_arr_ctrl =   linear_interp(t_arr2, t_arr1, v_arr)
    tau_arr_ctrl = linear_interp(t_arr2, t_arr1, tau_arr)

    print(q_arr.shape)
    print(v_arr.shape)
    print(tau_arr.shape)
    print(q_arr_ctrl.shape)
    print(v_arr_ctrl.shape)
    print(tau_arr_ctrl.shape)

    import matplotlib.pyplot as plt

    labels = [str(i) for i in range(7)] 
    plt.figure('q_arr_ctrl')
    plt.plot(t_arr2, q_arr_ctrl, label=labels)
    plt.grid()
    plt.legend()

    plt.figure('v_arr_ctrl')
    plt.plot(t_arr2, v_arr_ctrl, label=labels)
    plt.grid()
    plt.legend()

    plt.figure('tau_arr_ctrl')
    plt.plot(t_arr2, tau_arr_ctrl, label=labels)
    plt.grid()
    plt.legend()


    df_q = pd.DataFrame(q_arr_ctrl)
    df_v = pd.DataFrame(v_arr_ctrl)
    df_tau = pd.DataFrame(tau_arr_ctrl)


    traj_dir = os.path.join(SAVE_DIR, f'croco_panda_traj_{LABEL}')
    if not os.path.exists(traj_dir):
        os.makedirs(traj_dir)

    df_q.to_csv(  os.path.join(traj_dir, 'q.csv'),   sep=',', header=False, index=False)
    df_v.to_csv(  os.path.join(traj_dir, 'v.csv'),   sep=',', header=False, index=False)
    df_tau.to_csv(os.path.join(traj_dir, 'tau.csv'), sep=',', header=False, index=False)

    print('Saved ', traj_dir)

if GVIEWER:
    import crocoddyl

    # setup visualizer (instead of simulator)
    viz = pin.visualize.GepettoVisualizer(robot.model, robot.collision_model, robot.visual_model)
    viz.initViewer(loadModel=False)

    viz.viewer.gui.addSphere("world/target", 0.05, [0, 1, 0, 0.5])
    viz.viewer.gui.applyConfiguration("world/target", oMe_goal.translation.tolist() + [0, 0, 0, 1])
    viz.viewer.gui.addSphere("world/final", 0.05, [0, 0, 1, 0.5])

    viz.viewer.gui.applyConfiguration("world/final", oMe_fin.translation.tolist() + [0, 0, 0, 1])

    # Display trajectory solution in Gepetto Viewer
    display = crocoddyl.GepettoDisplay(robot)
    display.displayFromSolver(ocp.ddp, factor=1)

    print("Final - goal placement")
    print('translation (mm): ', 1e3*(oMe_fin.translation - oMe_goal.translation))
    print('orientation (deg): ', np.rad2deg(pin.log(oMe_goal.rotation.T*oMe_fin.rotation)))

if PLOT:
    fig_d, axes_d = ocp_utils.plot_ocp_results(
        ddp_data,
        which_plots="all",
        labels=None,
        markers=["."],
        colors=["b"],
        sampling_plot=1,
        SHOW=False,
    )

    ocp_utils.plot_ocp_state(ddp_data, fig_d['x'], axes_d['x'])


