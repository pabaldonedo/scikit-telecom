import numpy as np


class MultiLayer(object):
    def __init__(self):
        self.layers = list()

    def add_layer(self, layer):
        if isinstance(layer, Layer):
            self.layers.append(layer)


class Layer(object):
    def __init__(self, eps_r, mu_r, length):
        self.eps_r = eps_r
        self.mu_r = mu_r
        self.length = length


def brewster(n1, n2):
    """
    Calculates Brewster and critical angles. It deals with both isotropic and birefringent cases.

    In birefringent (uniaxial or biaxial) case, n1 and n2 are turned into 3-d arrays. thc_te is critical or maximum
    depending on n1[1]>n2[2] or n1[1])<n2[1] whereas thc_tm depends on n1[2]>n2[2] or n1[2]<n2[2]

    Parameters
    ---------
    n1 : int of float or ndarray
        refractive index of left media
    n2 : int of float or ndarray
        refractive index of right media

    Returns
    -------
    thb : float
        Brewster angle (in degrees)
    thc_te : float
        critical angle of reflection or maximum angle of refraction for TE in degrees
    thc_tm : float
        critical angle of reflection or maximum angle of refraction for TM in degrees

    """

    na, nb = __setup_medium_indexes(n1, n2)

    # calculate brewster angle
    if na[2] == nb[2]:
        thb = None
    else:
        thb = float(np.rad2deg(
            np.arctan((na[2] * nb[2] / na[0] ** 2) * np.sqrt((na[0] ** 2 - nb[0] ** 2) / (na[2] ** 2 - nb[2] ** 2)))))

    # calculate critical angle for TM
    if na[2] > nb[2]:
        thc_tm = float(np.rad2deg(
            np.arcsin(na[2] * nb[2] / np.sqrt(na[2] ** 2 * nb[2] ** 2 + na[0] ** 2 * (na[2] ** 2 - nb[2] ** 2)))))
    else:
        thc_tm = float(np.rad2deg(
            np.arcsin(na[2] * nb[2] / np.sqrt(na[2] ** 2 * nb[2] ** 2 + nb[0] ** 2 * (nb[2] ** 2 - na[2] ** 2)))))

    # calculate critical angle for TE
    if na[1] > nb[1]:
        thc_te = float(np.rad2deg(np.arcsin(nb[1] / na[1])))
    else:
        thc_te = float(np.rad2deg(np.arcsin(na[1] / nb[1])))

    # return results
    return thb, thc_te, thc_tm


def fresnel(n1, n2, th1):
    """
    Calculates Fresnel reflection coefficients for isotropic or birefringent media.

    It admits n1 and n2 indexes as 1d, 2d or 3d arrays according to cases: isotropic, birefringent uniaxial
    and birefringent biaxial.

    The function assumes that the interface is the x-y plane and that the plane of  incidence is the x-z plane,
    with the x,y,z axes being the diagonal optical axes where x,y are ordinary axes and z, extraordinary.

    TE or s-polarization has E = [0,Ey,0], and ordinary indices n1(1) or n2(1) on either side.

    TM or p-polarization has E = [Ex,0,Ez], and theta-dependent refractive index.

    Parameters
    ---------
    n1 : int or float or ndarray
        refractive index of left media
    n2 : int or float or ndarray
        refractive index of right media
    th1 : int or float or ndarray
        array of incident angles from medium a (in degrees) at which to evaluate rho's

    Returns
    -------
    rte : float or ndarray
        reflection coefficients for TE
    rtm : float or ndarray
        reflection coefficients for TM

    """
    na, nb = __setup_medium_indexes(n1, n2)
    th1 = np.deg2rad(th1)

    n = 1 / np.sqrt(np.cos(th1) ** 2 / na[0] ** 2 + np.sin(th1) ** 2 / na[2] ** 2)

    xe = (na[1] * np.sin(th1)) ** 2
    xm = (n * np.sin(th1)) ** 2

    rte = (na[1] * np.cos(th1) - np.sqrt(nb[1] ** 2 - xe)) / (na[1] * np.cos(th1) + np.sqrt(nb[1] ** 2 - xe))

    if na[2] == nb[2]:
        rtm = (na[0] - nb[0]) / (na[0] + nb[0] * np.ones(shape=th1.shape))
    else:
        rtm = (na[0] * na[2] * np.sqrt(nb[2] ** 2 - xm) - nb[0] * nb[2] * np.sqrt(na[2] ** 2 - xm)) / (
            na[0] * na[2] * np.sqrt(nb[2] ** 2 - xm) + nb[0] * nb[2] * np.sqrt(na[2] ** 2 - xm))

    if isinstance(th1, (int, float)):
        return float(rte), float(rtm)
    else:
        return rte, rtm


def snel(n1, n2, th1):
    """
    Calculates refraction angles using Snel's law for birefringent media

    Both media angles are given in degrees.

    Parameters
    ---------
    n1 : int or float or ndarray
        refractive index of left media
    n2 : int or float or ndarray
        refractive index of right media
    th1 : int or float or ndarray
        array of incident angles from medium a (in degrees) at which to evaluate rho's

    Returns
    -------
    th2_te : float or ndarray
        angle of refraction in second media for component TE
    th2_tm : float or ndarray
        angle of refraction in second media for component TM
    """
    na, nb = __setup_medium_indexes(n1, n2)
    th1 = np.deg2rad(th1)

    th2_te = np.rad2deg(np.arcsin(na[1] * np.sin(th1) / nb[1]))
    th2_tm = None

    if (isinstance(n1, (list, tuple, np.ndarray)) and len(n1) > 1) or \
            (isinstance(n2, (list, tuple, np.ndarray)) and len(n2) > 1):
        a = nb[0] ** 2 * nb[2] ** 2 * (na[0] ** 2 - na[2] ** 2) - na[0] ** 2 * na[2] ** 2 * (nb[0] ** 2 - nb[2] ** 2)
        b = nb[0] ** 2 * nb[2] ** 2 * na[2] ** 2

        th2_tm = np.rad2deg(np.arcsin(na[0] * na[2] * nb[2] * np.sin(th1) / np.sqrt(a * np.sin(th1) ** 2 + b)))

    if isinstance(th1, float) or isinstance(th1, int):
        if th2_tm is None:
            return float(th2_te)
        else:
            return th2_te, th2_tm
    else:
        if th2_tm is None:
            return th2_te
        else:
            return th2_te, th2_tm


def reflection(refractive_indices, layers_len, theta=0):
    """
    Calculates the reflection response and the wave impedance in a isotropic or birefringent multilayer structure. It
    will return a function for later evaluation.

    Parameters
    ---------
    refractive_indices : list or ndarray
        refractive indices of each layer.
    layers_len : list or ndarray
        length in lambdas of each layer
    theta : int or float
        incidence angle

    Returns
    -------
    reflection : function
        the returned function have to be evaluated in order to get reflection response and transverse wave impedance.
    """
    if isinstance(layers_len, (list, tuple)):
        layers_len = np.array(layers_len)

    # number of layers
    m = len(layers_len)

    # theta to radians
    theta = np.deg2rad(theta)

    # build general refractive indices matrix, valid for isotropic and birefringence
    is_birefringence = False  # flag for knowing if we have to calculate TM
    n = np.ones(shape=(3, m + 2))
    for i, ni in enumerate(refractive_indices):
        if isinstance(ni, (int, float)):
            n[:, i] = [ni, ni, ni]
        elif len(ni) == 2:
            n[:, i] = [ni[0]] + list(ni)
            is_birefringence = True
        else:
            n[:, i] = ni
            is_birefringence = True

    if is_birefringence:  # we have to calculate TM component
        na = (n[0, 0] * n[2, 0] * np.sin(theta)) ** 2 / (
             (n[2, 0] * np.cos(theta)) ** 2 + (n[0, 0] * np.sin(theta)) ** 2)
        c_tm = np.conj(np.sqrt(np.conj(1 - na / n[2, :] ** 2)))
        nt = c_tm / n[0, :]
        r_tm = -refractive_index_to_reflection_coeff(nt)

    na = (n[1, 0] * np.sin(theta)) ** 2
    c_te = np.conj(np.sqrt(np.conj(1 - na / n[1, :] ** 2)))
    nt = n[1, :] * c_te
    r_te = refractive_index_to_reflection_coeff(nt)

    if m > 0:
        layers_len_te = layers_len * c_te[1:m + 1]
        if is_birefringence:
            layers_len_tm = layers_len * c_tm[1:m + 1]

    # function to be returned for evaluation
    def f(x, r, pol='te'):
        gamma = r[m] * np.ones(shape=(1, 11))
        for k in range(m - 1, -1, -1):
            if pol == 'te':
                delta = 2 * np.pi * layers_len_te[k] / x
            else:
                delta = 2 * np.pi * layers_len_tm[k] / x
            z = np.exp(-2j * delta)
            gamma = (r[k] + gamma * z) / (1 + r[k] * gamma * z)
        gamma = gamma.flatten()
        z = (1 + gamma) / (1 - gamma)

        return gamma, z

    # if birefringence media return gamma and z for TE and TM
    def f_birefringence(x):
        g_te, z_te = f(x, r_te)
        g_tm, z_tm = f(x, r_tm, 'tm')
        return g_te, g_tm, z_te, z_tm

    # isotropic media TE and TM are equal, only returning TE
    def f_isotropic(x):
        return f(x, r_te)

    if is_birefringence:
        return f_birefringence
    else:
        return f_isotropic


def refractive_index_to_reflection_coeff(n):
    """
    Converts refractive indices to reflection coefficients of M-layer structure.

    Parameters
    ---------
    n : list or ndarray
        refractive indices of each layer.

    Returns
    -------
    c : ndarray
        reflection coefficients of each layer.
    """
    return -np.diff(n) / (2 * n.flatten(order='f')[:-1] + np.diff(n))


def __setup_medium_indexes(n1, n2):
    # if n1 and n2 are int or float number convert them to numpy arrays
    if isinstance(n1, (int, float)):
        n1 = np.array([n1])

    if isinstance(n2, (int, float)):
        n2 = np.array([n2])

    # check if isotropic or uniaxial case and create na and nb arrays
    if len(n1) == 1:
        na = np.array([n1, n1, n1])
    elif len(n1) == 2:
        na = np.array([n1[0], n1[0], n1[1]])
    else:
        na = n1

    if len(n2) == 1:
        nb = np.array([n2, n2, n2])
    elif len(n2) == 2:
        nb = np.array([n2[0], n2[0], n2[1]])
    else:
        nb = n2

    return na, nb