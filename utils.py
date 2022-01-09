import numpy as np

G = 6.6743015e-11  # from https://en.wikipedia.org/wiki/Gravitational_constant


def calculate_interaction(star_i: np.array, star_j: np.array) -> np.array:
    d = star_i[1:] - star_j[1:]
    r = np.dot(d, d)
    if r == 0:
        return 0  # fixes RuntimeWarning: invalid value encountered in true_divide
    r3 = r * np.sqrt(r)
    return G * d / r3


def calculate_interactions(stars_own: np.array, stars_buffer: np.array) -> np.array:
    n = stars_own.shape[0]
    m = stars_buffer.shape[0]
    other = not stars_own is stars_buffer

    interactions = np.zeros((n, m, 3))  # ix, iy, iz

    for i in range(n):
        for j in range(m):
            if other or i != j:
                interactions[i, j] = calculate_interaction(stars_own[i], stars_buffer[j])

    return interactions
