import numpy as np

from animals.crab.crab_controller import CrabController


class ZeroCrabController(CrabController):

    def step(self, action = None):
        CrabController.step(self, np.zeros(self.action_space.shape))
