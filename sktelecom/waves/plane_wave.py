import numpy as np

try:
    from matplotlib import pyplot as plt
    from matplotlib import animation
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    plt = Axes3D = animation = None

from sktelecom.constants import LIGHT_SPEED


class UniformPlaneWaveSSS(object):
    def __init__(self, phasor, eps_r=1, mu_r=1):
        if not is_plane_wave(phasor):
            raise TypeError("phasor is not a wave plane")

        self.phasor = phasor
        self.a = phasor.a
        self.g = phasor.g
        self.beta = np.imag(self.g)
        self.alpha = np.real(self.g)
        self.eps_r = eps_r
        self.mu_r = mu_r
        self.eta = 120 * np.pi * np.sqrt(mu_r / eps_r)
        self.k_prop = - self.beta / np.linalg.norm(self.beta)
        self.n = np.sqrt(eps_r * mu_r)

    def wavelength(self):
        return 2 * np.pi / np.linalg.norm(self.beta)

    def frequency(self):
        speed = LIGHT_SPEED / self.n
        return speed / self.wavelength()

    @staticmethod
    def decompose_linear(phasor):
        if not isinstance(phasor, Phasor):
            raise TypeError("argument must be a Phasor object")

        a = phasor.a
        if a[0] != 0:
            u1 = np.array([1, 0, 0])
        elif a[1] != 0:
            u1 = np.array([0, 1, 0])
        else:
            u1 = np.array([0, 0, 1])
        u2 = np.cross(phasor.k_prop, u1)

        al1 = np.dot(u1, a)
        al2 = np.dot(u2, a)

        return al1, al2, u1, u2

    @staticmethod
    def decompose_circular(phasor):
        if not isinstance(phasor, Phasor):
            raise TypeError("argument must be a Phasor object")

        al1, al2, u1, u2 = UniformPlaneWaveSSS.decompose_linear(phasor)

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

        def field_eval(kr, t):
            ax = np.abs(self.a[0]) * np.exp(-alpha * kr) * np.cos(omega * t - beta * kr + np.angle(self.a[0]))
            ay = np.abs(self.a[1]) * np.exp(-alpha * kr) * np.cos(omega * t - beta * kr + np.angle(self.a[1]))
            az = np.abs(self.a[2]) * np.exp(-alpha * kr) * np.cos(omega * t - beta * kr + np.angle(self.a[2]))
            return ax, ay, az

        def f(r, t):
            if isinstance(t, (int, float)) and isinstance(r, (list, np.ndarray)):
                ret = np.zeros(shape=(len(r), 3))
                for i, kr in enumerate(r):
                    ax, ay, az = field_eval(kr, t)
                    ret[i, :] = [ax, ay, az] + self.k_prop * kr
                return ret[:, 0], ret[:, 1], ret[:, 2]

            if isinstance(t, (list, np.ndarray)) and isinstance(r, (list, np.ndarray)):
                ret = np.zeros(shape=(len(r), 3, len(t)))
                for i, kr in enumerate(r):
                    for j, tr in enumerate(t):
                        ax, ay, az = field_eval(kr, tr)

                        ret[i, :, j] = [ax, ay, az] + self.k_prop * kr

                return ret[:, 0, :], ret[:, 1, :], ret[:, 2, :]

        return f

    def plot(self, r, t, plot_speed=1):
        if not (plt and Axes3D and animation):
            # matplotlib is not installed
            raise ImportError("matplotlib is not installed")

        at = self.time_domain()
        x, y, z = at(r, t)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlim((np.min(x), np.max(x)))
        ax.set_ylim((np.min(y), np.max(y)))
        ax.set_zlim((np.min(z), np.max(z)))
        line, = ax.plot([], [], [], lw=2)

        def init():
            line.set_3d_properties([])
            return line.set_data([], []),

        def animate(i):
            line.set_data(x[:, i], y[:, i])
            line.set_3d_properties(z[:, i])
            return line,

        interval = 1000. / (len(t) / (t[-1] - t[0]))

        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(t), interval=interval, blit=False)
        plt.show()


class Phasor(object):
    def __init__(self, a, gamma):
        self.a = a
        self.g = gamma
        self.beta = np.imag(self.g)
        self.alpha = np.real(self.g)
        self.k_prop = self.beta / np.linalg.norm(self.beta)

    @staticmethod
    def alpha(gamma):
        return np.real(gamma)

    @staticmethod
    def beta(gamma):
        return np.imag(gamma)

    @staticmethod
    def dir_propagation(beta):
        return beta / np.linalg.norm(beta)


class ElectricField(UniformPlaneWaveSSS):
    def __init__(self, phasor, eps_r=1, mu_r=1, **kwargs):
        super(ElectricField, self).__init__(phasor, eps_r, mu_r)

    @classmethod
    def from_time_domain(cls, e_mod, e_angle, k, **kwargs):
        if 'beta' in kwargs.keys():
            beta = kwargs['beta']

        if 'alpha' in kwargs.keys():
            alpha = kwargs['beta']
        else:
            alpha = 0

        if ('freq' or 'omega') and 'eps_r' in kwargs.keys():
            if 'mu_r' not in kwargs.keys():
                kwargs['mu_r'] = 1

            if 'freq' in kwargs.keys():
                omega = 2 * np.pi * kwargs['freq']
            else:
                omega = kwargs['omega']
            beta = np.sqrt(kwargs['eps_r'] * kwargs['mu_r']) * omega / LIGHT_SPEED

            alpha = 0

        e = e_mod * np.exp(1j * e_angle)
        gamma = alpha * k - 1j * beta * k

        return cls(Phasor(e, gamma), **kwargs)

    def magnetic_field(self):
        a = 1 / self.eta * np.cross(self.k_prop, self.a)
        return MagneticField(Phasor(a, self.g), eps_r=self.eps_r, mu_r=self.mu_r)


class MagneticField(UniformPlaneWaveSSS):
    def __init__(self, phasor, eps_r=1, mu_r=1):
        super(MagneticField, self).__init__(phasor, eps_r, mu_r)

    @classmethod
    def from_time_domain(cls, h_mod, h_angle, alpha, beta, k, **kwargs):
        h = h_mod * np.exp(1j * h_angle)
        gamma = alpha * k - 1j * beta * k

        return cls(Phasor(h, gamma, **kwargs))

    def electric_field(self):
        a = self.eta * np.cross(self.a, self.k_prop)
        return ElectricField(Phasor(a, self.g), eps_r=self.eps_r, mu_r=self.mu_r)


class ElectromagneticWave(object):
    @classmethod
    def from_electric_wave(cls, wave):
        pass

    @classmethod
    def from_magnetic_wave(cls, wave):
        pass


def is_plane_wave(phasor):
    if not isinstance(phasor, Phasor):
        raise TypeError("argument must be a Phasor object")

    # get two vectors from wave
    _, _, u1, u2 = UniformPlaneWaveSSS.decompose_linear(phasor)

    if np.dot(np.cross(u1, u2), phasor.k_prop) == 1.0:
        return True
    else:
        return False