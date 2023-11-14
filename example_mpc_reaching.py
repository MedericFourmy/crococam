
import time
import numpy as np
import pinocchio as pin
from pinbullet import SimuProxy
from gviewer_mpc import GviewerMpc
np.set_printoptions(precision=4, linewidth=180)
from ocp_def import OCP, ConfigOCP
from robot_utils import create_panda
import log_and_plot


GVIEWER_REPLAY = True

PLOT = True

# Goal
RANDOM_REF_EVERY = 3000
DELTA_TRANS = 0.9*np.array([0.2, 0.2, 0.1])
DELTA_ORIENTATION = 30*np.array([1, 1, 1])  # degs

# Simulation
SEED = 0
N_SIM = 20000
SIM_SLEEP = False
DISABLE_JOINT_LIMITS = True
DT_SIM = 1/1000
SIGMA_TAU = 0.05  # control noise

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
cfg.diag_ee_vel = np.array(3*[1]+3*[10])
cfg.T = 50
cfg.dt = 2e-2  # seconds
ocp = OCP(robot.model, cfg)

# Solve every...
DT_DDP_SOLVE = 10*cfg.dt  # seconds
PRINT_EVERY = 500
SOLVE_EVERY = int(DT_DDP_SOLVE/DT_SIM)
MAX_ITER = 1
print('SOLVE_EVERY', SOLVE_EVERY)

ee_fid = robot.model.getFrameId(cfg.ee_name)
oMe0 = robot.framePlacement(robot.q0, ee_fid)  # FK
oMe_goal = oMe0.copy()

x0 = np.concatenate([robot.q0, np.zeros(robot.model.nv)])
xs_init, us_init = ocp.quasistatic_init(x0)
ocp.set_ee_placement_ref(oMe_goal)
ocp.set_state_reg_ref(x0)
ocp.set_initial_state(x0)
# Initial solution
ocp.safe_solve(xs_init, us_init, maxiter=100, is_feasible=False)

fixed_joints = ['panda_finger_joint1', 'panda_finger_joint2']

# Simulation
sim = SimuProxy()
sim.init(DT_SIM, 'panda', fixed_joints, visual=True, disable_joint_limits=DISABLE_JOINT_LIMITS)
sim.setState(x0)

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
        delta_trans = np.random.normal(0, DELTA_TRANS)
        delta_aa = np.random.normal(0, np.deg2rad(DELTA_ORIENTATION))
        oMe_goal.translation = oMe0.translation + delta_trans
        oMe_goal.rotation = oMe0.rotation @ pin.exp3(delta_aa)

    if (k % PRINT_EVERY) == 0:
        print(f'{k}/{N_SIM}')

    #  Warm start using previous solution
    if (k % SOLVE_EVERY) == 0:
        # Set fixed initial state of the tracjectory
        ocp.set_initial_state(xk)
        ocp.set_ee_placement_ref(oMe_goal)

        # Shift the result trajectory according to solve frequency and ddp integration dt
        shift = int(DT_DDP_SOLVE / cfg.dt)
        xs_init = list(ocp.ddp.xs[shift:]) + shift*[ocp.ddp.xs[-1]]
        xs_init[0] = xk
        us_init = list(ocp.ddp.us[shift:]) + shift*[ocp.ddp.us[-1]]

        # Solve
        t_solve1 = time.time()
        ocp.safe_solve(xs_init, us_init, maxiter=MAX_ITER, is_feasible=False)
        
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
    
    # Compute Ricatti feedback
    u_ricatti = ocp.ddp.us[0] + ocp.ddp.K[0] @ (ocp.ddp.xs[0] - xk)
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