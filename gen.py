import numpy as np

if __name__ == '__main__':
    stars = np.random.random((840, 4))  # M, x, y, z
    print(stars)
    np.save('stars.npy', stars)
