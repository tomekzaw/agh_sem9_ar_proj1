import numpy as np

G = 6.6743015e-11  # from https://en.wikipedia.org/wiki/Gravitational_constant


def calculate_interaction(star_i: np.array, star_j: np.array) -> np.array:
    d = star_i[1:] - star_j[1:]
    r = np.dot(d, d)
    r3 = r * np.sqrt(r) + 1e-10  # fixes RuntimeWarning: invalid value encountered in true_divide
    return G * star_j[0] * d / r3


def calculate_interactions_own(stars_own: np.array) -> np.array:
    n = stars_own.shape[0]
    interactions = np.zeros((n, 3))  # ax, ay, az

    for i in range(n):
        for j in range(n):
            if i != j:
                interactions[i] += calculate_interaction(stars_own[i], stars_own[j])

    return interactions


def calculate_interactions_own_with_buffer(stars_own: np.array, stars_buffer: np.array) -> np.array:
    n = stars_own.shape[0]
    interactions = np.zeros((n, 3))  # ax, ay, az

    for i in range(n):
        for j in range(n):
            interactions[i] += calculate_interaction(stars_own[i], stars_buffer[j])

    return interactions


# TODO: perform calculations using numpy vector functions
