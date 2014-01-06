import numpy as np


def brewster(n1, n2):

    brewster_angle = np.rad2deg(np.arctan(n2/n1))
    critic_angle = np.rad2deg(np.arcsin(n2/n1))

    return brewster_angle, critic_angle