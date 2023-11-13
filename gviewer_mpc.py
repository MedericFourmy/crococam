import numpy as np
import pinocchio as pin
from example_robot_data import load
from pinbullet import freezed_robot

class GviewerMpc:

    def __init__(self, robot_name: str, fixed_joints: list[str] = None, nb_keyframes: int = 3, color_start=[1, 0, 0, 0.2], color_end=[0, 1, 0, 0.2]) -> None:
        assert nb_keyframes > 0 
        self.nb_keyframes = nb_keyframes

        robot = load(robot_name)

        if fixed_joints is not None:
            robot = freezed_robot(robot, fixed_joints)

        self.robots = []
        self.vizs = []
        for i in range(nb_keyframes):
            # We need to load the panda model from example_robot_data to be able to color it
            # in GepettoViewer (not sure why) -> harcode panda for now
            r = pin.RobotWrapper(robot.model, robot.collision_model, robot.visual_model)
            r.q0 = robot.q0
            self.robots.append(r)
            viz = pin.visualize.GepettoVisualizer(
                model=r.model,
                collision_model=r.collision_model,
                visual_model=r.visual_model,
            )
            viz.initViewer(loadModel=True, sceneName=f'world/preview_{i}')
            self.vizs.append(viz)

        # all viz variables point to the smae gui object, why not use the last one
        self.gv = viz.viewer.gui

        # change color and transparency of preview robots using there name in the scene graph
        colors = np.linspace(color_start, color_end, nb_keyframes)
        for node in self.gv.getNodeList():
            for i in range(nb_keyframes):
                if f'/preview_{i}' in node:
                    self.gv.setColor(node, colors[i].tolist())
        self.display(self.nb_keyframes*[r.q0])
        self.gv.refresh()

    def display(self, q_lst):
        # Select q at relugar intervals of the traj and call viz.display on them

        # Hardcoded !!
        # Panda from example_robot_data has 2 more degrees of freedom than
        #  -> hack: append 2 zero valued coordinates
        for viz, q in zip(self.vizs, q_lst):
            viz.display(q)

    def display_keyframes(self, qs):
        """
        Display nb_keyframes keyframes along a long joint trajectory qs.

        We only want to display sparse moments of the predicted trajectory
        -> select keyframes with equal time separation, including the last 
        moment, excluding the current state (assumed to be handled by the user)
        """
        assert(qs.shape[1] == self.robots[0].nq)

        indices = (len(qs)/self.nb_keyframes)*np.arange(self.nb_keyframes)
        indices = indices.astype(np.int64)

        # 2 inversions for reasons evoked above:
        # - the first to select last moment (and in between ones) but not first
        # Example:
        # For ls(qs) == 100 and nb_keyframes == 5,
        # indices = [0,20,40,60,80]
        # -> [80,60,40,20,0]
        # - second inversion to have forward timestep configs
        q_arr_preview = qs[indices[::-1] + self.nb_keyframes - 1][::-1]

        self.display(q_arr_preview)


if __name__ == '__main__':
    from copy import deepcopy

    robot_name = 'panda'
    fixed_joints = ['panda_finger_joint1', 'panda_finger_joint2']
    # fixed_joints = None
    nb = 5

    gmpc = GviewerMpc(robot_name, nb, fixed_joints)
    robot = gmpc.robots[0]
    q_lst = []
    for i in range(nb):
        q = deepcopy(robot.q0)
        q[1] = i*np.pi/5
        q_lst.append(q)
    # gmpc.display(q_lst)

    # qs
    ls = 113
    qs = np.linspace(np.zeros(robot.model.nq)-0.3, robot.q0, ls)
    gmpc.display_keyframes(qs)

