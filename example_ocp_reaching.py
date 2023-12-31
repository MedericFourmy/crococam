
import numpy as np
np.set_printoptions(precision=4, linewidth=180)
import pinocchio as pin

from crococam import ocp_utils
from crococam.ocp_def import OCP, ConfigOCP
from crococam.robot_utils import create_panda

GVIEWER = True
PLOT = True

# DELTA_TRANS = np.array([-0.20, -0.3, 0.1])
DELTA_TRANS = np.array([-0.80, -0.35, -0.5])  # puts itself in limit if no joint limit cost

robot = create_panda()

# Def the OCP
cfg = ConfigOCP
cfg.ee_name = 'panda_hand'
cfg.w_frame_running = 10
cfg.w_joint_limits_running = 1000.0
cfg.w_joint_limits_terminal = 1000.0
cfg.w_frame_vel_running = 0.0
cfg.w_frame_vel_terminal = 0.0 
cfg.T = 100
cfg.dt = 1e-2  # seconds
ocp = OCP(robot.model, cfg)

# Set references
oMe_goal = robot.framePlacement(robot.q0, robot.model.getFrameId(cfg.ee_name))
oMe_goal.translation += DELTA_TRANS
x0 = np.concatenate([robot.q0, np.zeros(robot.model.nv)])
xs_init, us_init = ocp.quasistatic_init(x0)
ocp.set_ee_placement_ref(oMe_goal)
ocp.set_state_reg_ref(x0)
ocp.set_initial_state(x0)
ocp.ddp.solve(xs_init, us_init, maxiter=50, is_feasible=False)
print("Iteration #:", ocp.ddp.iter)


if GVIEWER:
    import crocoddyl

    # setup visualizer (instead of simulator)
    viz = pin.visualize.GepettoVisualizer(robot.model, robot.collision_model, robot.visual_model)
    viz.initViewer(loadModel=False)

    # solution joint trajectory
    xs = np.array(ocp.ddp.xs)
    q_final = xs[-1, : robot.model.nq]
    oMe_fin = robot.framePlacement(q_final, robot.model.getFrameId(cfg.ee_name))

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
    # Extract DDP data and plot
    ddp_data = ocp_utils.extract_ocp_data(ocp.ddp, cfg.ee_name)

    fig_d, axes_d = ocp_utils.plot_ocp_results(
        ddp_data,
        which_plots="all",
        labels=None,
        markers=["."],
        colors=["b"],
        sampling_plot=1,
        SHOW=False,
        x_limits_lower=ocp.x_limits_lower,
        x_limits_upper=ocp.x_limits_upper,
    )

    ocp_utils.plot_ocp_state(ddp_data, fig_d['x'], axes_d['x'])


