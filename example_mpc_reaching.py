
import time
import numpy as np
import pinocchio as pin
from example_robot_data import load

from pinbullet import SimuProxy, freezed_robot
from gviewer_mpc import GviewerMpc

np.set_printoptions(precision=4, linewidth=180)

from ocp_def import OCP, ConfigOCP
from robot_utils import create_panda

GVIEWER_REPLAY = True
PLOT = True

DELTA_TRANS = np.array([-0.30, -0.3, 0.1])
# DELTA_TRANS = np.zeros(3)

robot = create_panda()


# Simulation
SIM_SLEEP = False
DISABLE_JOINT_LIMITS = False
N_SIM = 10000
# DT_SIM = 1/240  # pybullet default
DT_SIM = 1/1000
SIGMA_TAU = 0.0  # control noise


# Def the OCP
cfg = ConfigOCP
cfg.ee_name = 'panda_hand'
cfg.w_frame_terminal = 10
cfg.w_frame_running = 1
cfg.w_joint_limits_running = 1000.0
cfg.w_joint_limits_terminal = 1000.0

cfg.T = 50
cfg.dt = 2e-2  # seconds
ocp = OCP(robot.model, cfg)

# Solve every...
DT_DDP_SOLVE = 4*cfg.dt  # seconds
PRINT_EVERY = 500
SOLVE_EVERY = int(DT_DDP_SOLVE/DT_SIM)
MAX_ITER = 30
print('SOLVE_EVERY', SOLVE_EVERY)

ee_fid = robot.model.getFrameId(cfg.ee_name)
oMe0 = robot.framePlacement(robot.q0, ee_fid)  # FK
oMe_goal = oMe0.copy()
oMe_goal.translation += DELTA_TRANS

x0 = np.concatenate([robot.q0, np.zeros(robot.model.nv)])
xs_init, us_init = ocp.quasistatic_init(x0)
ocp.set_ee_placement_ref(oMe_goal)
ocp.set_state_reg_ref(x0)
# Initial solution
success = ocp.ddp.solve(xs_init, us_init, maxiter=1, is_feasible=False)

fixed_joints = ['panda_finger_joint1', 'panda_finger_joint2']

# Simulation
sim = SimuProxy()
sim.init(DT_SIM, 'panda', fixed_joints, visual=True, disable_joint_limits=DISABLE_JOINT_LIMITS)
sim.setState(x0)

# Visualization of mpc preview
gmpc = GviewerMpc('panda', fixed_joints, nb_keyframes=4)

# Force disturbance
t1_fext, t2_fext = 1.0, 4.0
# fext = np.array([0,30,0, 0,0,0])
fext = np.array([0,20,0, 0,0,0])
# fext = np.array([0,0,0, 0,0,0])
frame_fext = "panda_hand"


# Logs
t_solve = []
dt_solve = []
nb_iter_solve = []
q_sim_arr = np.zeros((N_SIM, robot.nq))
v_sim_arr = np.zeros((N_SIM, robot.nv))
u_ref_arr = np.zeros((N_SIM, robot.nv))
u_ricatti_arr = np.zeros((N_SIM, robot.nv))
u_noisy_arr = np.zeros((N_SIM, robot.nv))
t_sim_arr = DT_SIM*np.arange(N_SIM)
print('\n==========================')
print('Begin simulation + control')
print('Apply force between ', t1_fext, t2_fext, ' seconds')
print('   -> ', t1_fext/DT_SIM, t2_fext/DT_SIM, ' iterations')
print('fext: ', fext)
for k in range(N_SIM):
    t1 = time.time()
    xk = sim.getState()
    qk_sim, vk_sim = xk[:robot.nq], xk[robot.nq:]

    xk_sim = np.concatenate([qk_sim, vk_sim])

    tk = DT_SIM*k 

    if (k % PRINT_EVERY) == 0:
        print(f'{k}/{N_SIM}')

    #  Warm start using previous solution
    if (k % SOLVE_EVERY) == 0:
        # Set fixed initial state of the tracjectory
        ocp.ddp.problem.x0 = xk_sim

        # shift the result trajectory according to solve frequency and ddp integration dt
        shift = int(DT_DDP_SOLVE / cfg.dt)
        xs_init = list(ocp.ddp.xs[shift:]) + shift*[ocp.ddp.xs[-1]]
        xs_init[0] = xk_sim
        us_init = list(ocp.ddp.us[shift:]) + shift*[ocp.ddp.us[-1]]

        # Solve
        t1 = time.time()
        success = ocp.ddp.solve(xs_init, us_init, maxiter=MAX_ITER, is_feasible=False)
        t_solve.append(tk)
        dt_solve.append(1e3*(time.time() - t1))  # store milliseconds
        nb_iter_solve.append(ocp.ddp.iter)

    gmpc.display_keyframes(np.array(ocp.ddp.xs)[:,:robot.nq])

    # Simulate external force
    if t1_fext < tk < t2_fext:
        sim.applyExternalForce(fext, frame_fext, rf_frame=pin.LOCAL_WORLD_ALIGNED)
    
    # Compute Ricatti feedback
    u_ricatti = ocp.ddp.us[0] + ocp.ddp.K[0] @ (ocp.ddp.xs[0] - xk_sim)
    u_noisy = u_ricatti + np.random.normal(0, SIGMA_TAU)
    sim.step(u_noisy)

    delay = time.time() - t1
    if SIM_SLEEP and delay < DT_SIM: 
        time.sleep(DT_SIM - delay)

    # Logs
    t_sim_arr[k] = tk
    q_sim_arr[k,:] = qk_sim
    v_sim_arr[k,:] = vk_sim
    u_ref_arr[k,:] = ocp.ddp.us[0]
    u_ricatti_arr[k,:] = u_ricatti
    u_noisy_arr[k,:] = u_noisy


if GVIEWER_REPLAY:
    print('\n=================')
    print('Realtime playback')
    # setup visualizer 
    robot.initViewer(loadModel=True)

    gui = robot.viewer.gui
    gui.addXYZaxis('world/target', [1,1,1,1], 0.01, 0.05)
    gui.applyConfiguration("world/target", list(pin.SE3ToXYZQUAT(oMe_goal)))
    gui.addXYZaxis('world/ee_current', [1,1,1,1], 0.01, 0.05)
    # Viewer loop 
    k = 0
    while k < N_SIM:
        t1 = time.time()
        robot.display(q_sim_arr[k,:])
        oMe = robot.framePlacement(q_sim_arr[k,:], ee_fid)
        gui.applyConfiguration("world/ee_current", list(pin.SE3ToXYZQUAT(oMe)))
        delay = time.time() - t1
        if delay < DT_SIM: 
            time.sleep(DT_SIM - delay)
        k += 1

print(f'# DOF, T, DT_OCP, Mean Solve: {9-len(fixed_joints)}, {cfg.T}, {cfg.dt}, {np.mean(dt_solve)}')

if PLOT:
    print('\n=========')
    print('Plot traj')
    import matplotlib.pyplot as plt

    ##############################
    # State
    fig, axes = plt.subplots(robot.nq,2)
    fig.canvas.manager.set_window_title('sim_state')
    fig.suptitle('State trajectories (q,v)', size=18)
    for i in range(robot.nq):
        axes[i,0].plot(t_sim_arr, q_sim_arr[:,i])
        axes[i,1].plot(t_sim_arr, v_sim_arr[:,i])
    axes[-1,0].set_xlabel('Time (s)', fontsize=16)
    axes[-1,1].set_xlabel('Time (s)', fontsize=16)


    ##############################
    # Controls
    fig, axes = plt.subplots(robot.nq,1)
    fig.canvas.manager.set_window_title('joint_torques')
    fig.suptitle('Joint torques', size=12)
    for i in range(robot.nq):
        axes[i].plot(t_sim_arr, u_ref_arr[:,i], 'b.', label='u_ref')
        axes[i].plot(t_sim_arr, u_ricatti_arr[:,i], 'g.', label='u_ricatti')
        axes[i].plot(t_sim_arr, u_noisy_arr[:,i], 'r.', label='u_noisy')
    plt.legend()
    axes[-1].set_xlabel('Time (s)', fontsize=16)

    ##############################
    # Solve time
    fig, axes = plt.subplots(2,1)
    fig.canvas.manager.set_window_title('solve_times')
    axes[0].plot(t_solve, dt_solve, '.')
    axes[0].set_xlabel('Solve times (ms)', fontsize=16)
    axes[1].plot(t_solve, nb_iter_solve, '.')
    axes[1].set_xlabel('# iterations', fontsize=16)
    plt.grid()

    ##############################
    # End effector pose trajectory
    oMe_lst = [robot.framePlacement(q, ee_fid, True) 
               for q in q_sim_arr]

    t_oe_arr = np.array([M.translation for M in oMe_lst])
    o_oe_arr = np.rad2deg(np.array([pin.log3(M.rotation) for M in oMe_lst]))

    fig, axes = plt.subplots(3,2)
    fig.canvas.manager.set_window_title('end_effector_traj')
    fig.suptitle('End effector trajectories (position,orientation)', size=18)
    for i in range(3):
        l = 'xyz'[i]
        c = 'rgb'[i]
        axes[i,0].plot(t_sim_arr, t_oe_arr[:,i], f'.{c}', label=f't{l}')
        axes[i,0].plot([t_sim_arr[0], t_sim_arr[-1]], 2*[oMe_goal.translation[i]], ':k', label=f'ref_t_{l}')
        axes[i,1].plot(t_sim_arr, o_oe_arr[:,i])

    plt.grid()
    plt.legend()
    axes[-1,0].set_xlabel('Time (s)', fontsize=16)
    axes[-1,1].set_xlabel('Time (s)', fontsize=16)

    # o_nu_e_lst = [robot.frameVelocity(q, v, ee_fid, True, pin.LOCAL_WORLD_ALIGNED) 
    #             for q, v in zip(q_sim_arr, v_sim_arr)]



    plt.show()