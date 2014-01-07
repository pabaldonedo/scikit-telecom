import numpy as np


def brewster(n1, n2):

    # if n1 and n2 are int or float number convert them to numpy arrays
    if isinstance(n1, int) or isinstance(n1, float):
        n1 = np.array([n1])

    if isinstance(n2, int) or isinstance(n2, float):
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

