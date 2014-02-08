import numpy as np

from sktelecom.constants import LIGHT_SPEED


class PlaneWave(object):
    def __init__(self, a, k):
        self.a = a
        self.k = k
        self.beta = np.imag(k)
        self.alpha = np.real(k)
        self.k_prop = self.beta / np.linalg.norm(self.beta)

    def wavelength(self):
        return 2 * np.pi / np.linalg.norm(self.k)

    def frequency(self):
        return LIGHT_SPEED / self.wavelength()