
import time
import numpy as np
np.set_printoptions(precision=4, linewidth=180)
import pinocchio as pin

from crococam.pinbullet import SimuProxy
from crococam.pin_utils import perturb_placement
from crococam.gviewer_mpc import GviewerMpc
from crococam.ocp_def import OCP, ConfigOCP, get_warm_start_traj
from crococam.robot_utils import create_panda
from crococam import log_and_plot


GVIEWER_REPLAY = True
PLOT = True

# Goal
RANDOM_REF_EVERY = 3000
SIGMA_TRANS = 0.9*np.array([0.2, 0.2, 0.1])
SIGMA_ORIENTATION = 30*np.array([1, 1, 1])  # degs

# Simulation
SEED = 0
N_SIM = 12000
SIM_SLEEP = False
DISABLE_JOINT_LIMITS = True
DT_SIM = 1/1000
SIGMA_TAU = 0.01  # control noise

np.random.seed(SEED)

# Force disturbance
t1_fext, t2_fext = 1.0, 4.0
# fext = np.array([0,30,0, 0,0,0])
# fext = np.array([0,20,0, 0,0,0])
fext = np.array([0,0,0, 0,0,0])
frame_fext = "panda_hand"

robot = create_panda()

# Def the OCP
cfg = ConfigOCP
cfg.ee_name = 'panda_hand'
cfg.w_frame_terminal = 100
cfg.w_frame_running = 10
cfg.w_joint_limits_running = 100.0
cfg.w_joint_limits_terminal = 100.0
cfg.w_frame_vel_running = 0.1
cfg.w_frame_vel_terminal = 0.1
cfg.diag_velocity = np.array(3*[1]+3*[10])
cfg.T = 50
cfg.dt = 2e-2  # seconds
ocp = OCP(robot.model, cfg)

# MPC setting
DT_DDP_SOLVE = 20*cfg.dt  # solve at lower frequency than integration step
APPLY_ZERO_SHIFT = True
SOLVE_EVERY = int(DT_DDP_SOLVE/DT_SIM)
MAX_ITER = 1

# Viz
PRINT_EVERY = 500
print('SOLVE_EVERY', SOLVE_EVERY)

ee_fid = robot.model.getFrameId(cfg.ee_name)
oMe0 = robot.framePlacement(robot.q0, ee_fid)  # FK
oMe_goal = oMe0.copy()

x_init = np.concatenate([robot.q0, np.zeros(robot.model.nv)])
xs_warm, us_warm = ocp.quasistatic_init(x_init)
ocp.set_ee_placement_ref(oMe_goal)
ocp.set_state_reg_ref(x_init)
ocp.set_initial_state(x_init)
# Initial solution
ocp.safe_solve(xs_warm, us_warm, maxiter=100, is_feasible=False)

fixed_joints = ['panda_finger_joint1', 'panda_finger_joint2']

# Simulation
sim = SimuProxy()
sim.init(DT_SIM, 'panda', fixed_joints, visual=True, disable_joint_limits=DISABLE_JOINT_LIMITS)
sim.setState(x_init)

# Visualization of mpc preview
# gmpc = GviewerMpc('panda', fixed_joints, nb_keyframes=4)


# Logs
logs = log_and_plot.LoggerMPC(DT_SIM)
print('\n==========================')
print('Begin simulation + control')
print('Apply force between ', t1_fext, t2_fext, ' seconds')
print('   -> ', t1_fext/DT_SIM, t2_fext/DT_SIM, ' iterations')
print('fext: ', fext)
for k in range(N_SIM):
    t1 = time.time()
    xk = sim.getState()
    qk, vk = xk[:robot.nq], xk[robot.nq:]

    tk = DT_SIM*k 

    if (k % RANDOM_REF_EVERY == 0):
        oMe_goal = perturb_placement(oMe0, SIGMA_TRANS, SIGMA_ORIENTATION)

    if (k % PRINT_EVERY) == 0:
        print(f'{k}/{N_SIM}')

    #  Warm start using previous solution
    if (k % SOLVE_EVERY) == 0:
        # Set fixed initial state of the tracjectory
        ocp.set_initial_state(xk)
        ocp.set_ee_placement_ref(oMe_goal)

        # Shift the result trajectory according to solve frequency and ddp integration dt
        xs_warm, us_warm = get_warm_start_traj(ocp.ddp, xk, cfg.dt, DT_DDP_SOLVE, APPLY_ZERO_SHIFT)

        # Solve
        t_solve1 = time.time()
        ocp.safe_solve(xs_warm, us_warm, maxiter=MAX_ITER, is_feasible=False)
        u0, K0, x0 = ocp.get_ricatti_feedback()
        
        # Log
        logs.append({
            't_solve': tk,
            'dt_solve': time.time() - t_solve1,
            'nb_iter_solve': ocp.ddp.iter,
        })

    # gmpc.display_keyframes(np.array(ocp.ddp.xs)[:,:robot.nq])

    # Simulate external force
    if t1_fext < tk < t2_fext:
        sim.applyExternalForce(fext, frame_fext, rf_frame=pin.LOCAL_WORLD_ALIGNED)
    
    u_ricatti = u0 + K0 @ (x0 - xk)
    u_noisy = u_ricatti + np.random.normal(0, SIGMA_TAU*np.ones(robot.nv))
    sim.step(u_noisy)

    delay = time.time() - t1
    if SIM_SLEEP and delay < DT_SIM: 
        time.sleep(DT_SIM - delay)

    # Logs
    logs.append({
        't': tk,
        'q': qk,
        'v': vk,
        'u_ref': ocp.ddp.us[0],
        'u_cmd': u_noisy,
        'pose_oe_goal': pin.SE3ToXYZQUAT(oMe_goal),
    })

logs.arrayify()

if GVIEWER_REPLAY:
    print('\n=================')
    print('Realtime playback')
    # setup visualizer 
    robot.initViewer(loadModel=True)

    gui = robot.viewer.gui
    gui.addXYZaxis('world/goal', [1,1,1,1], 0.01, 0.05)
    gui.addXYZaxis('world/ee_current', [1,1,1,0.7], 0.01, 0.05)

    # Viewer loop 
    k = 0
    while k < N_SIM:
        t1 = time.time()
        robot.display(logs['q'][k,:])
        oMe = robot.framePlacement(logs['q'][k,:], ee_fid)
        gui.applyConfiguration("world/ee_current", list(pin.SE3ToXYZQUAT(oMe)))
        gui.applyConfiguration("world/goal", list(logs['pose_oe_goal'][k,:]))
        delay = time.time() - t1
        if delay < DT_SIM: 
            time.sleep(DT_SIM - delay)
        k += 1

if PLOT:
    import matplotlib.pyplot as plt
    log_and_plot.plot_end_effector_vel(logs, robot, ee_fid)
    log_and_plot.plot_solver(logs)
    log_and_plot.plot_controls(logs, robot.nv)
    log_and_plot.plot_states(logs, robot.nq, x_lower=ocp.x_limits_lower, x_upper=ocp.x_limits_upper)
    log_and_plot.plot_end_effector_pose(logs, robot, ee_fid)
    plt.show()