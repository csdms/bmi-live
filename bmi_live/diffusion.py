"""A model of temperature diffusion over a rectangular plate."""
import numpy as np
import yaml


class Diffusion(object):
    """Model of temperature diffusion on a plate."""

    def __init__(self, config_file=None):
        """Initialize the model."""

        if config_file is not None:
            with open(config_file, 'r') as fp:
                parameters = yaml.load(fp)
            for key, value in parameters.items():
                setattr(self, key, value)
        else:
            self.nx = 8
            self.ny = 6
            self.dx = 1.0
            self.dy = 1.0
            self.alpha = 0.9

        self.time = 0.0
        self.dt = min(self.dx, self.dy) ** 2.0 / (4.0 * self.alpha)
        self.dt /= 2.0

        self.temperature = np.zeros((self.ny, self.nx))
        self.new_temperature = self.temperature.copy()

    def advance(self):
        """Advance the model by one time step."""

        self.solve()
        self.time += self.dt

    def solve(self):
        """Solve the diffusion equation."""

        dx2, dy2 = self.dx**2, self.dy**2
        coef = self.alpha * self.dt / (2.0*(dx2 + dy2))

        for i in range(1, self.ny-1):
            for j in range(1, self.nx-1):
                self.new_temperature[i,j] = \
                    self.temperature[i,j] + coef * (
                    dx2*(self.temperature[i,j-1] + self.temperature[i,j+1]) +
                    dy2*(self.temperature[i-1,j] + self.temperature[i+1,j]) -
                    2.0*(dx2 + dy2)*self.temperature[i,j])

        self.new_temperature[(0, -1), :] = 0.0
        self.new_temperature[:, (0, -1)] = 0.0

        self.temperature[:] = self.new_temperature
