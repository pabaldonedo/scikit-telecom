
from numpy.testing import assert_approx_equal
from sktelecom import multilayer


def test_brewster():
    n1 = 1.2
    n2 = 1.5

    thb, thc = multilayer.brewster(n1, n2)

    assert_approx_equal(thb, 51.3402, significant=5)
    assert_approx_equal(thc, 53.1301, significant=5)

