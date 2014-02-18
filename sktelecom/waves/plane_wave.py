import numpy as np

from sktelecom.constants import LIGHT_SPEED


class UniformPlaneWaveSSS(object):
    def __init__(self, phasor):
        self.phasor = phasor
        self.a = phasor.a
        self.g = phasor.g
        self.beta = np.imag(self.g)
        self.alpha = np.real(self.g)
        self.k_prop = self.beta / np.linalg.norm(self.beta)

    def wavelength(self):
        return 2 * np.pi / np.linalg.norm(self.beta)

    def frequency(self):
        return LIGHT_SPEED / self.wavelength()

    def decompose_linear(self):
        if self.a[0] != 0:
            u1 = np.array([1, 0, 0])
        elif self.a[1] != 0:
            u1 = np.array([0, 1, 0])
        else:
            u1 = np.array([0, 0, 1])
        u2 = np.cross(self.k_prop, u1)

        al1 = np.dot(u1, self.a)
        al2 = np.dot(u2, self.a)

        return al1, al2, u1, u2

    def decompose_circular(self):
        al1, al2, u1, u2 = self.decompose_linear()

        v1 = u1 + 1j * u2
        v2 = u1 - 1j * u2
        v1 /= np.linalg.norm(v1)
        v2 /= np.linalg.norm(v2)

        m = np.array([[1, 1j], [1, -1j]])
        ac = np.dot(np.array([[al1, al2]]), np.linalg.inv(m)).flatten()

        return ac[0], ac[1], v1, v2

    def time_domain(self):
        alpha = np.linalg.norm(self.alpha)
        beta = np.linalg.norm(self.beta)
        omega = 2 * np.pi * self.frequency()

        def f(r, t):
            ret = np.zeros(shape=(len(r), 3))
            for i, kr in enumerate(r):
                ax = np.abs(self.a[0]) * np.exp(-alpha * kr) * np.cos(omega * t - beta * kr + np.angle(self.a[0]))
                ay = np.abs(self.a[1]) * np.exp(-alpha * kr) * np.cos(omega * t - beta * kr + np.angle(self.a[1]))
                az = np.abs(self.a[2]) * np.exp(-alpha * kr) * np.cos(omega * t - beta * kr + np.angle(self.a[2]))
                ret[i, :] = [ax, ay, az] + self.k_prop * kr
            return ret

        return f


class Phasor(object):
    def __init__(self, a, gamma):
        self.a = a
        self.g = gamma


class ElectricalField(UniformPlaneWaveSSS):
    def __init__(self, phasor):
        super(ElectricalField, self).__init__(phasor)