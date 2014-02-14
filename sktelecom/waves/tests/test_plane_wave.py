import numpy as np
from numpy.testing import assert_approx_equal, assert_array_almost_equal

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


def test_plane_decompose_linear():
    a = np.array([-20j, 5 * np.sqrt(3), 15])
    k = np.array([0, np.sqrt(3), 1]) * 30j * np.pi

    p = waves.PlaneWave(a, k)
    al1, al2, u1, u2 = p.decompose_linear()

    assert_array_almost_equal(al1, -20j)
    assert_array_almost_equal(al2, -8.66025403784 + 0j)
    assert_array_almost_equal(u1, np.array([1, 0, 0]))
    assert_array_almost_equal(u2, np.array([0, 0.5, -0.8660254]))


def test_plane_decompose_circular():
    a = np.array([-20j, 5 * np.sqrt(3), 15])
    k = np.array([0, np.sqrt(3), 1]) * 30j * np.pi

    p = waves.PlaneWave(a, k)
    ac1, ac2, v1, v2 = p.decompose_circular()

    assert_array_almost_equal(ac1, -5.66987298108j)
    assert_array_almost_equal(ac2, -14.3301270189j)
    assert_array_almost_equal(v1, np.array([0.70710678 + 0.j, 0 + 0.35355339j, 0 - 0.61237244j]))
    assert_array_almost_equal(v2, np.array([0.70710678 + 0.j, 0 - 0.35355339j, 0 + 0.61237244j]))