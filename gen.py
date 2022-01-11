import numpy as np

if __name__ == '__main__':
    nstars = 4000

    stars = np.zeros((nstars, 7))  # M, x, y, z, vx, vy, vz

    # M
    stars[:, 0] = np.random.uniform(low=2e2, high=4e2, size=nstars)

    # x, y, z
    stars[:, 1:4] = np.random.random((nstars, 3))

    # vx, vy, vz
    stars[:, 4:] = np.random.uniform(low=-2e-4, high=2e-4, size=(nstars, 3))

    np.save('stars.npy', stars)
