import numpy as np


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


def fresnel(n1, n2, theta):
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
    theta : int or float or ndarray
        array of incident angles from medium a (in degrees) at which to evaluate rho's

    Returns
    -------
    rte : float or ndarray
        reflection coefficients for TE
    rtm : float or ndarray
        reflection coefficients for TM

    """
    na, nb = __setup_medium_indexes(n1, n2)
    theta = np.deg2rad(theta)

    n = 1 / np.sqrt(np.cos(theta) ** 2 / na[0] ** 2 + np.sin(theta) ** 2 / na[2] ** 2)

    xe = (na[1] * np.sin(theta)) ** 2
    xm = (n * np.sin(theta)) ** 2

    rte = (na[1] * np.cos(theta) - np.sqrt(nb[1] ** 2 - xe)) / (na[1] * np.cos(theta) + np.sqrt(nb[1] ** 2 - xe))

    if na[2] == nb[2]:
        rtm = (na[0] - nb[0]) / (na[0] + nb[0] * np.ones(shape=theta.shape))
    else:
        rtm = (na[0] * na[2] * np.sqrt(nb[2] ** 2 - xm) - nb[0] * nb[2] * np.sqrt(na[2] ** 2 - xm)) / (
            na[0] * na[2] * np.sqrt(nb[2] ** 2 - xm) + nb[0] * nb[2] * np.sqrt(na[2] ** 2 - xm))

    if isinstance(theta, int) or isinstance(theta, float):
        return float(rte), float(rtm)
    else:
        return rte, rtm


def snel(n1, n2, th1):
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