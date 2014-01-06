import numpy as np
from numpy.testing import assert_array_almost_equal
from sktelecom import multilayer


def test_brewster():
    n1 = 1.5
    n2 = 1

    thb, thc = multilayer.brewster(n1, n2)

    assert_array_almost_equal(np.array([thb, thc]), np.array([33.6901, 41.8103]), decimal=4)
