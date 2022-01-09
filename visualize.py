import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import seaborn as sns

plt.style.use('dark_background')

plt.rcParams['grid.color'] = 'gray'

fps = 25
dpi = 300

transparent = (1.0, 1.0, 1.0, 0.0)

if __name__ == '__main__':
    snapshots = np.load('snapshots.npy')

    steps, nstars, _ = snapshots.shape

    coords = snapshots.reshape(-1, 3).T
    mins, maxs = np.min(coords, axis=1), np.max(coords, axis=1)
    c, d = (mins + maxs) / 2, np.max(maxs - mins)
    xlim, ylim, zlim = np.array([c - d / 2, c + d / 2]).T

    colors = sns.color_palette('hls', nstars)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set(xlim=xlim, ylim=ylim, zlim=zlim)
    ax.xaxis.set_pane_color(transparent)
    ax.yaxis.set_pane_color(transparent)
    ax.zaxis.set_pane_color(transparent)
    fig.tight_layout()

    def animate(step):
        print(step)
        coords = snapshots[step, :, :].T
        scatter = ax.scatter(*coords, color=colors, marker='.')
        return scatter,

    ani = FuncAnimation(fig, animate, interval=1000 / fps, frames=steps)
    ani.save('zad3.gif', dpi=dpi, writer=PillowWriter(fps=fps))
