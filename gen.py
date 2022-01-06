import numpy as np

if __name__ == '__main__':
    stars = np.random.random((10000, 4))  # M, x, y, z
    np.save('stars.npy', stars)
