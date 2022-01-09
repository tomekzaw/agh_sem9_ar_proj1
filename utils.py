import numpy as np

G = 6.6743015e-11  # from https://en.wikipedia.org/wiki/Gravitational_constant


def calculate_interaction(star_i: np.array, star_j: np.array) -> np.array:
    d = star_j[1:] - star_i[1:]
    r = np.dot(d, d)
    if r == 0:
        return 0  # fixes RuntimeWarning: invalid value encountered in true_divide
    r3 = r * np.sqrt(r)
    return G * d / r3


def calculate_interactions(stars_own: np.array, stars_buffer: np.array, same: bool, use_symmetry: bool = False) -> np.array:
    n = stars_own.shape[0]
    m = stars_buffer.shape[0]

    interactions = np.zeros((n, m, 3))  # ix, iy, iz

    if same:
        if use_symmetry:
            for i in range(n):
                for j in range(i):
                    interactions[i, j] = calculate_interaction(stars_own[i], stars_buffer[j])
                    interactions[j, i] = -interactions[i, j]
        else:
            for i in range(n):
                for j in range(n):
                    if i != j:
                        interactions[i, j] = calculate_interaction(stars_own[i], stars_buffer[j])
    else:
        for i in range(n):
            for j in range(m):
                interactions[i, j] = calculate_interaction(stars_own[i], stars_buffer[j])

    return interactions
