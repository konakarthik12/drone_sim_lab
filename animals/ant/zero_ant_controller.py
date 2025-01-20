import numpy as np

from animals.ant.ant_controller import AntController


class ZeroAntController(AntController):

    def step(self, action = None):
        AntController.step(self, np.zeros(self.action_space.shape))
