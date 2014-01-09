import numpy as np
from numpy.testing import assert_array_almost_equal
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