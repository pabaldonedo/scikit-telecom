import numpy as np
from numpy.testing import assert_approx_equal, assert_array_almost_equal
from nose.tools import assert_raises, assert_almost_equal

from sktelecom import waves


def test_plane_wave_frequency():
    a = np.array([-1, 0, 1j])
    k = np.array([0, -0.68j * np.pi, 0])
    phasor = waves.Phasor(a, k)

    p = waves.UniformPlaneWaveSSS(phasor)
    freq = p.frequency()

    assert_approx_equal(freq, 101929435.72)


def test_plane_wave_wavelength():
    a = np.array([-1, 0, 1j])
    k = np.array([0, -0.68j * np.pi, 0])
    phasor = waves.Phasor(a, k)

    p = waves.UniformPlaneWaveSSS(phasor)
    wave_len = p.wavelength()

    assert_approx_equal(wave_len, 2.9411764705882346)


def test_plane_decompose_linear():
    a = np.array([-20j, 5 * np.sqrt(3), 15])
    k = np.array([0, np.sqrt(3), 1]) * 30j * np.pi
    phasor = waves.Phasor(a, k)

    al1, al2, u1, u2 = waves.UniformPlaneWaveSSS.decompose_linear(phasor)

    assert_array_almost_equal(al1, -20j)
    assert_array_almost_equal(al2, -8.66025403784 + 0j)
    assert_array_almost_equal(u1, np.array([1, 0, 0]))
    assert_array_almost_equal(u2, np.array([0, 0.5, -0.8660254]))


def test_plane_decompose_circular():
    a = np.array([-20j, 5 * np.sqrt(3), 15])
    k = np.array([0, np.sqrt(3), 1]) * 30j * np.pi
    phasor = waves.Phasor(a, k)

    ac1, ac2, v1, v2 = waves.UniformPlaneWaveSSS.decompose_circular(phasor)

    assert_array_almost_equal(ac1, -5.66987298108j)
    assert_array_almost_equal(ac2, -14.3301270189j)
    assert_array_almost_equal(v1, np.array([0.70710678 + 0.j, 0 + 0.35355339j, 0 - 0.61237244j]))
    assert_array_almost_equal(v2, np.array([0.70710678 + 0.j, 0 - 0.35355339j, 0 + 0.61237244j]))


def atest_electrical_field_uniform_plane_wave_sss():
    a = (10 + 3j) * np.array([0, 1, 1])
    gamma = np.array([-1 * np.pi * 1j, 0, 0])

    phasor = waves.Phasor(a, gamma)
    e = waves.ElectricalField(phasor)

    et = e.time_domain()
    r = np.linspace(0, 5 * np.pi, 10)

    x, y, z = et(r, 0)

    assert_array_almost_equal(x, np.array(
        [0., -1.74532925, -3.4906585, -5.23598776, -6.98131701, -8.72664626, -10.47197551, -12.21730476, -13.96263402,
         -15.70796327]))
    assert_array_almost_equal(y, np.array(
        [10., 4.81433413, -3.29213784, -9.40130426, -9.80679784, -4.26260989, 3.86765885, 9.65145946, 9.5798207,
         3.69620505]))
    assert_array_almost_equal(z, np.array(
        [10., 4.81433413, -3.29213784, -9.40130426, -9.80679784, -4.26260989, 3.86765885, 9.65145946, 9.5798207,
         3.69620505]))


def test_electrical_field_uniform_plane_wave_sss_time_dependency():
    a = (10 + 3j) * np.array([0, 1, 1])
    gamma = np.array([-1.3333333e-8 * np.pi * 1j, 0, 0])

    phasor = waves.Phasor(a, gamma)
    e = waves.ElectricalField(phasor)

    et = e.time_domain()
    r = np.linspace(0, e.wavelength(), 100)
    t = np.linspace(0, 5, 100)


def test_is_plane_wave_valid():
    a = np.array([1, -1j, 0])
    gamma = np.array([0, 0, -np.pi * 2j])

    phasor = waves.Phasor(a, gamma)

    assert waves.is_plane_wave(phasor)


def test_is_plane_wave_invalid():
    a = np.array([1, -1j, 2j])
    gamma = np.array([-2j, 4j, -np.pi * 2j])

    phasor = waves.Phasor(a, gamma)
    assert not waves.is_plane_wave(phasor)


def test_uniform_plane_wave_sss_valid_phasor():
    a = np.array([1, -1j, 2j])
    gamma = np.array([-2j, 4j, -np.pi * 2j])

    phasor = waves.Phasor(a, gamma)

    assert_raises(TypeError, waves.UniformPlaneWaveSSS, phasor)


def test_electric_wave_time_domain_dir_prop():
    # creates an electric wave in the time domain and checks its propagation

    e_mod = np.array([0, 0, 10])
    e_angle = np.array([0, 0, 0])
    k = np.array([-1, 0, 0])
    omega = 1.5 * np.pi * 1e6
    eps_r = 4

    e = waves.ElectricalField.from_time_domain(e_mod, e_angle, k, omega=omega, eps_r=eps_r)

    assert_array_almost_equal(e.k_prop, np.array([-1, 0, 0]), decimal=4)
    assert_almost_equal(e.frequency(), 750000.0)
    assert_almost_equal(e.wavelength(), 199.862, places=3)
    assert_almost_equal(np.linalg.norm(e.beta), np.pi / 100, places=4)

    h = e.magnetic_field()

    assert_array_almost_equal(h.a, np.array([0 + 0j, 1 / (6 * np.pi) + 0j, 0 + 0j]), decimal=4)


def test_magnetic_phasor():
    h_field = np.array([-1, 0, 1j]) * 2.92e-3
    gamma = np.array([0, -0.68j * np.pi, 0])

    phasor = waves.Phasor(h_field, gamma)

    h = waves.MagneticField(phasor)

    assert_almost_equal(h.frequency(), 101929435.72)
    assert_almost_equal(h.wavelength(), 2.9412, places=2)

    e = h.electric_field()

    assert_array_almost_equal(e.a, -0.3504 * np.pi * np.array([1j, 0, 1]), decimal=4)
