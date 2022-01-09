import numpy as np

if __name__ == '__main__':
    stars = np.zeros((840, 7))  # M, x, y, z, vx, vy, vz

    # M
    stars[:, 0] = np.random.uniform(low=2e2, high=4e2, size=840)

    # x, y, z
    stars[:, 1:4] = np.random.random((840, 3))

    # vx, vy, vz
    stars[:, 4:] = np.random.uniform(low=-2e-4, high=2e-4, size=(840, 3))

    np.save('stars.npy', stars)
