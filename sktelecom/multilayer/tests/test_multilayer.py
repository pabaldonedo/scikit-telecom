import numpy as np
from numpy.testing import assert_array_almost_equal, assert_approx_equal
from sktelecom import multilayer


def test_brewster_glass2air():
    n1 = 1.5
    n2 = 1

    thb, thc, _ = multilayer.brewster(n1, n2)

    assert_array_almost_equal(np.array([thb, thc]), np.array([33.6901, 41.8103]), decimal=4)


def test_brewster_air2glass():
    n1 = 1
    n2 = 1.5

    thb, thc, _ = multilayer.brewster(n1, n2)

    assert_array_almost_equal(np.array([thb, thc]), np.array([56.3099, 41.8103]), decimal=4)


def test_brewster_birefringent_uniaxial():
    n1 = np.array([1.1, 1.2])
    n2 = [1, 1]

    thb, thc_te, thc_tm = multilayer.brewster(n1, n2)

    assert_array_almost_equal(np.array([thb, thc_te, thc_tm]), np.array([34.4166, 65.3800, 58.6984]), decimal=4)


def test_brewster_birefringent_biaxial():
    n1 = [1.1, 1.2, 1.4]
    n2 = [1.5, 1.5, 1.5]

    thb, thc_te, thc_tm = multilayer.brewster(n1, n2)

    assert_array_almost_equal(np.array([thb, thc_te, thc_tm]), np.array([73.0770, 53.1301, 68.9605]), decimal=4)


def test_fresnel_air2glass_30deg():
    n1 = 1
    n2 = 1.5
    theta = 30

    rte, rtm = multilayer.fresnel(n1, n2, theta)

    assert_array_almost_equal(np.array([rte, rtm]), np.array([-0.2404, -0.1589]), decimal=4)


def test_fresnel_glass2air_30deg():
    n1 = 1.5
    n2 = 1
    theta = 30

    rte, rtm = multilayer.fresnel(n1, n2, theta)

    assert_array_almost_equal(np.array([rte, rtm]), np.array([0.3252, 0.0679]), decimal=4)


def test_fresnel_air2glass_0_to_90deg():
    n1 = 1
    n2 = 1.5
    theta = np.arange(100, step=10, dtype=np.int)

    rte, rtm = multilayer.fresnel(n1, n2, theta)

    rte_test = np.array([-0.2000, -0.2041, -0.2170, -0.2404, -0.2778, -0.3347, -0.4202, -0.5474, -0.7339, -1.0000])
    rtm_test = np.array([-0.2000, -0.1959, -0.1829, -0.1589, -0.1196, -0.0572, 0.0424, 0.2061, 0.4866, 1.0000])

    assert_array_almost_equal(rte, rte_test, decimal=4)
    assert_array_almost_equal(rtm, rtm_test, decimal=4)


def test_fresnel_birefringent_biaxial_mediums_0_to_90deg():
    n1 = [1.2, 1.1, 1.3]
    n2 = [1, 1.5, 1.3]
    theta = np.arange(100, step=10, dtype=np.int)

    rte, rtm = multilayer.fresnel(n1, n2, theta)

    rte_test = np.array([-0.1538, -0.1573, -0.1683, -0.1886, -0.2218, -0.2741, -0.3562, -0.4858, -0.6890, -1.0000])
    rtm_test = np.array([0.0909, 0.0909, 0.0909, 0.0909, 0.0909, 0.0909, 0.0909, 0.0909, 0.0909, 0.0909])

    assert_array_almost_equal(rte, rte_test, decimal=4)
    assert_array_almost_equal(rtm, rtm_test, decimal=4)


def test_snel_air2glass():
    n1 = 1
    n2 = 1.5
    th1 = 30

    th2_test = multilayer.snel(n1, n2, th1)
    assert_approx_equal(th2_test, 19.4712, significant=4)


def test_snel_air2glass_0_to_90():
    n1 = 1
    n2 = 1.5
    th1 = np.arange(100, step=10, dtype=np.int)

    th2_test = multilayer.snel(n1, n2, th1)
    th2 = np.array([0, 6.6478, 13.1801, 19.4712, 25.3740, 30.7102, 35.2644, 38.7896, 41.0364, 41.8103])

    assert_array_almost_equal(th2, th2_test, decimal=4)


def test_snel_birefringent_biaxial_mediums_0_to_90deg():
    n1 = [1.2, 1.1, 1.3]
    n2 = [1, 1.5, 1.3]
    th1 = np.arange(100, step=10, dtype=np.int)

    th2_te_test, th2_tm_test = multilayer.snel(n1, n2, th1)
    th2_te = np.array([0, 7.3160, 14.5257, 21.5102, 28.1238, 34.1780, 39.4263, 43.5595, 46.2358, 47.1666])
    th2_tm = np.array([0, 11.9471, 23.5940, 34.7150, 45.1975, 55.0368, 64.3066, 73.1270, 81.6408, 90.0000])

    assert_array_almost_equal(th2_te_test, th2_te, decimal=4)
    assert_array_almost_equal(th2_tm_test, th2_tm, decimal=4)

