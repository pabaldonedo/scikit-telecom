import numpy as np
from numpy.testing import assert_approx_equal

from sktelecom import waves


def test_plane_wave_frequency():
    a = np.array([-1, 0, 1j])
    k = np.array([0, -0.68j * np.pi, 0])

    p = waves.PlaneWave(a, k)
    freq = p.frequency()

    assert_approx_equal(freq, 101929435.72)


def test_plane_wave_wavelength():
    a = np.array([-1, 0, 1j])
    k = np.array([0, -0.68j * np.pi, 0])

    p = waves.PlaneWave(a, k)
    wave_len = p.wavelength()

    assert_approx_equal(wave_len, 2.9411764705882346)