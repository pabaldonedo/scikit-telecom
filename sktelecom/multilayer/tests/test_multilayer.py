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


def test_reflection_isotropic_4layers():
    n = [1, 1.2, 1.4, 1.3, 1.5, 1.2]
    lengths = [0.5, 0.3, 0.6, 0.8]
    theta = 30
    x = np.linspace(0, 1, 11)

    gamma = np.array(
        [np.nan + np.nan * 1j, -0.137816332463059 + 0.107008569516682j, -0.225526159874376 - 0.167055607818688j,
         -0.194446662594797 + 0.0834325706719984j, -0.0147172622100718 + 0.0326501189942636j,
         -0.0998094771049842 - 0.111581468216867j, 0.00168962035234530 + 0.176254253975789j,
         0.0466197225764525 + 0.0995766214269103j, -0.0676732258765700 + 0.106717670487398j,
         -0.392192618389167 + 0.00679432003501674j, -0.0570194530030898 - 0.163688238485510j])

    z = np.array([np.nan + np.nan * 1j, 0.742342100088068 + 0.163862594005407j, 0.602181435778377 - 0.218398759939331j,
                  0.666285558588159 + 0.116390703729396j, 0.968953868190505 + 0.0633541780403127j,
                  0.799969392148472 - 0.182616381930101j, 0.942825521196948 + 0.343010820907773j,
                  1.07516099671971 + 0.216741971554807j, 0.854702546797630 + 0.185384001812714j,
                  0.436548606872172 + 0.00701079064205943j, 0.847800488033844 - 0.286147247432013j])

    reflection_test = multilayer.reflection(n, lengths, theta)
    gamma_test, z_test = reflection_test(x)

    assert_array_almost_equal(gamma, gamma_test, decimal=2)
    assert_array_almost_equal(z, z_test, decimal=2)


def test_reflection_isotropic2layers():
    n = [1, 1.3, 1.8, 1.1]
    lengths = [0.8, 0.9]
    theta = 45
    x = np.linspace(0, 1, 11)

    gamma = np.array(
        [np.nan + np.nan * 1j, 0.283967663684886 + 0.160980378136825j, 0.142682828393841 - 0.222765981593735j,
         0.314052483101709 + 0.0555014309254255j, -0.399481181649957 + 0.147296497491893j,
         0.176129918046768 - 0.225077241815512j, 0.0736142292048855 + 0.252133690555673j,
         -0.469380508365036 - 0.332732330376140j, -0.126523702797059 + 0.145501839323489j,
         -0.167219815623956 - 0.273581598402077j, 0.208105560485227 + 0.219451639549374j])

    z = np.array([np.nan + np.nan * 1j, 1.65878110498675 + 0.597754550955714j, 1.18531262473555 - 0.567833385427666j,
                  1.89671088353181 + 0.234378862890625j, 0.413443290399082 + 0.148766020443193j,
                  1.25896791457287 - 0.617138889612772j, 1.01003247384705 + 0.547068966023542j,
                  0.294728221313601 - 0.293183376143170j, 0.746241852491948 + 0.225544656371076j,
                  0.624241623521170 - 0.380701744098275j, 1.34546496212529 + 0.649980737027770j])

    reflection_test = multilayer.reflection(n, lengths, theta)
    gamma_test, z_test = reflection_test(x)

    assert_array_almost_equal(gamma, gamma_test, decimal=2)
    assert_array_almost_equal(z, z_test, decimal=2)


def test_reflection_birefringent2layers():
    n = [[1, 1, 1], [1, 1.2, 1.3], [1.4, 1.5, 1.6], [1.1, 1.5, 1.2]]
    lengths = [0.6, 0.9]
    theta = 30
    x = np.linspace(0, 1, 11)

    gamma_te = np.array(
        [np.nan + np.nan * 1j, -0.221105340803292 - 0.0674218683131845j, 0.00887414723011978 + 0.0370764525357238j,
         -0.0317027567923196 - 0.0980791703258817j, -0.0985206172586343 - 0.126571801231585j,
         -0.169074723286103 + 0.114398775889770j, -0.168976127340151 - 0.114447043450823j,
         0.00541146260392240 - 0.0469762110344397j, -0.0318118681493969 + 0.0981695206644608j,
         - 0.146620547408592 + 0.122884798730675j, -0.221277132234859 + 0.0671456553688908j])

    gamma_tm = np.array(
        [np.nan + np.nan * 1j, -0.0603822561785445 - 0.0162021455939989j, 0.259331383496968 - 0.0844129193306986j,
         0.0139330060371999 - 0.203912137924608j, 0.105998131148818 - 0.177982724216180j,
         -0.0617924473044704 + 0.221758375388031j, -0.0885545699618609 - 0.0350109864189424j,
         0.256387406351406 - 0.0895320519564950j, 0.0507326956546752 + 0.115422541966280j,
         0.0804431930479251 + 0.0807684800717024j, -0.0481090113219318 + 0.184074074463138j])

    z_te = np.array(
        [np.nan + np.nan * 1j, 0.632882386585887 - 0.0901576445213884j, 1.01508731277772 + 0.0753812332699681j,
         0.921180337212945 - 0.182637656320786j, 0.796776897232744 - 0.207025052348655j,
         0.694528794546607 + 0.165816620567425j, 0.694655437738372 - 0.165912972883647j,
         1.00640583466003 - 0.0947661674811622j, 0.920949109874753 + 0.182764570905210j,
         0.724449715511821 + 0.184811493820482j, 0.632694656292792 + 0.0897653364830083j])

    z_tm = np.array(
        [np.nan + np.nan * 1j, 0.885672048058300 - 0.0288121881398498j, 1.66563938787815 - 0.303799023747039j,
         0.945081259801837 - 0.402229950451263j, 1.15184347575888 - 0.428400630073104j,
         0.804879340071796 + 0.376954188402672j, 0.835400170921513 - 0.0590316482339595j,
         1.65114022988460 - 0.319201190019317j, 1.07619281170163 + 0.252446756400182j,
         1.15830978492696 + 0.189573280883253j, 0.851102826674768 + 0.325099809156775j])

    reflection_test = multilayer.reflection(n, lengths, theta)
    gamma_te_test, gamma_tm_test, z_te_test, z_tm_test = reflection_test(x)

    assert_array_almost_equal(gamma_te, gamma_te_test, decimal=2)
    assert_array_almost_equal(z_te, z_te_test, decimal=2)

    assert_array_almost_equal(gamma_tm, gamma_tm_test, decimal=2)
    assert_array_almost_equal(z_tm, z_tm_test, decimal=2)


def test_refractive_index_to_reflection_coeff():
    n = np.arange(1, 2, 0.1)
    c_test = multilayer.refractive_index_to_reflection_coeff(n)
    c = np.array([-0.0476, -0.0435, -0.0400, -0.0370, -0.0345, -0.0323, -0.0303, -0.0286, -0.0270])

    assert_array_almost_equal(c, c_test, decimal=2)
