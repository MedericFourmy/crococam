import numpy as np
import matplotlib.pyplot as plt
import pinocchio as pin

class LoggerMPC:

    def __init__(self, dt_sys) -> None:
        self.dt_syst = dt_sys

        # Logs:
        self.logs = {
            # Solve frequency
            't_solve': [],
            'dt_solve': [],
            'nb_iter_solve': [],
            # Feedback frequency
            't': [],
            'q': [],
            'v': [],
            'u_ref': [],
            'u_cmd': [],
            'pose_oe_goal': [],
        }
    
    def __getitem__(self, key):
        return self.logs[key]

    def append(self, d: dict):
        for k, v in d.items():
            if k not in self.logs:
                print('LoggerMPC does not support {k} key')
            else:
                self.logs[k].append(v)

    def arrayify(self):
        for k, v in self.logs.items():
            self.logs[k] = np.array(v)


def plot_states(logs, nq, x_lower=None, x_upper=None):
    fig, axes = plt.subplots(nq,2)
    fig.canvas.manager.set_window_title('States')
    fig.suptitle('State trajectories (q,v)', size=18)
    # q_lower = x_lower[:nq]
    # q_upper = x_upper[:nq]
    # v_lower = x_lower[nq:]
    # v_upper = x_upper[nq:]
    for i in range(nq):
        axes[i,0].plot(logs['t'], logs['q'][:,i], '.')
        axes[i,1].plot(logs['t'], logs['v'][:,i], '.')

        if x_lower is not None:
            axes[i,0].hlines(x_lower[i],    logs['t'][0], logs['t'][-1], 'r', linestyles='dashed')
            axes[i,1].hlines(x_lower[nq+i], logs['t'][0], logs['t'][-1], 'r', linestyles='dashed')
        if x_upper is not None:
            axes[i,0].hlines(x_upper[i],    logs['t'][0], logs['t'][-1], 'r', linestyles='dashed')
            axes[i,1].hlines(x_upper[nq+i], logs['t'][0], logs['t'][-1], 'r', linestyles='dashed')

        axes[i,0].grid()
        axes[i,1].grid()
    axes[-1,0].set_xlabel('Time (s)', fontsize=16)
    axes[-1,1].set_xlabel('Time (s)', fontsize=16)


def plot_controls(logs, nv):
    fig, axes = plt.subplots(nv,1)
    fig.canvas.manager.set_window_title('Torques')
    fig.suptitle('Joint torques', size=12)
    for i in range(nv):
        axes[i].plot(logs['t'], logs['u_ref'][:,i], 'b.', label='u_ref')
        axes[i].plot(logs['t'], logs['u_cmd'][:,i], 'g.', label='u_cmd')
        axes[i].grid()
    axes[i].legend()
    axes[-1].set_xlabel('Time (s)', fontsize=16)


def plot_solver(logs):
    fig, axes = plt.subplots(2,1)
    fig.canvas.manager.set_window_title('Solver')
    axes[0].plot(logs['t_solve'], 1e3*logs['dt_solve'], '.')
    axes[0].set_xlabel('Solve times (ms)', fontsize=16)
    axes[1].plot(logs['t_solve'], logs['nb_iter_solve'], '.')
    axes[1].set_xlabel('# iterations', fontsize=16)
    axes[0].grid()
    axes[1].grid()


def plot_end_effector(logs, robot, ee_fid):
    oMe_lst = [robot.framePlacement(q, ee_fid, True) for q in logs['q']]
    oMe_goal_lst = [pin.XYZQUATToSE3(pose) for pose in logs['pose_oe_goal']]

    t_oe_arr = np.array([M.translation for M in oMe_lst])
    o_oe_arr = np.rad2deg(np.array([pin.log3(M.rotation) for M in oMe_lst]))
    t_oe_goal_arr = np.array([M.translation for M in oMe_goal_lst])
    o_oe_goal_arr = np.rad2deg(np.array([pin.log3(M.rotation) for M in oMe_goal_lst]))

    fig, axes = plt.subplots(3,2)
    fig.canvas.manager.set_window_title('End effector')
    fig.suptitle('End effector trajectories (position,orientation)', size=18)
    for i in range(3):
        l = 'xyz'[i]
        axes[i,0].plot(logs['t'], t_oe_arr[:,i], f'.b', label=f't{l}_c')
        axes[i,0].plot(logs['t'], t_oe_goal_arr[:,i], f'.k', label=f't{l}_g')
        axes[i,0].grid()
        axes[i,1].plot(logs['t'], o_oe_arr[:,i], f'.b', label=f'o{l}_c')
        axes[i,1].plot(logs['t'], o_oe_goal_arr[:,i], f'.k', label=f'o{l}_g')
        axes[i,1].grid()
    axes[-1,0].set_xlabel('Time (s)', fontsize=16)
    axes[-1,1].set_xlabel('Time (s)', fontsize=16)
    plt.legend()





        
        
        
        
        
        
        
        