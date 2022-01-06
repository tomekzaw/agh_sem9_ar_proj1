import numpy as np

G = 6.6743015e-11  # from https://en.wikipedia.org/wiki/Gravitational_constant


def calculate_interactions_own(stars_own: np.array) -> np.array:
    n = stars_own.shape[0]
    interactions = np.zeros((n, 3))  # ax, ay, az

    M = stars_own[:, 0]
    x = stars_own[:, 1]
    y = stars_own[:, 2]
    z = stars_own[:, 3]

    for i in range(n):
        for j in range(n):
            if i != j:
                xij = x[i] - x[j]
                yij = y[i] - y[j]
                zij = z[i] - z[j]
                rij2 = xij*xij + yij*yij + zij*zij
                rij = np.sqrt(rij2)
                rij3 = rij2 * rij
                Mj_rij3 = M[j] / rij3
                interactions[i, 0] += Mj_rij3 * xij
                interactions[i, 1] += Mj_rij3 * yij
                interactions[i, 2] += Mj_rij3 * zij

    interactions *= G

    return interactions


def calculate_interactions_own_with_buffer(stars_own: np.array, stars_buffer: np.array) -> np.array:
    n = stars_own.shape[0]
    interactions = np.zeros((n, 3))

    x_own = stars_own[:, 1]
    y_own = stars_own[:, 2]
    z_own = stars_own[:, 3]

    M_buffer = stars_buffer[:, 0]
    x_buffer = stars_buffer[:, 1]
    y_buffer = stars_buffer[:, 2]
    z_buffer = stars_buffer[:, 3]

    for i in range(n):
        for j in range(n):
            xij = x_own[i] - x_buffer[j]
            yij = y_own[i] - y_buffer[j]
            zij = z_own[i] - z_buffer[j]
            rij2 = xij*xij + yij*yij + zij*zij
            rij = np.sqrt(rij2)
            rij3 = rij2 * rij + 1e-12
            Mj_rij3 = M_buffer[j] / rij3
            interactions[i, 0] += Mj_rij3 * xij
            interactions[i, 1] += Mj_rij3 * yij
            interactions[i, 2] += Mj_rij3 * zij

    interactions *= G

    return interactions
