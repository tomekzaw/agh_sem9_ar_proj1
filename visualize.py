import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm

fps = 25
dpi = 150

transparent = (1.0, 1.0, 1.0, 0.0)

if __name__ == '__main__':
    snapshots = np.load('snapshots.npy')

    steps, nstars, _ = snapshots.shape

    coords = snapshots[:100, :, :].reshape(-1, 3).T
    mins, maxs = np.min(coords, axis=1), np.max(coords, axis=1)
    c, d = (mins + maxs) / 2, np.max(maxs - mins)
    xlim, ylim, zlim = np.array([c - d / 2, c + d / 2]).T

    palette = sns.color_palette('hls', nstars)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    fig.tight_layout()
    fig.patch.set_facecolor('white')
    ax.set(xlim=xlim, ylim=ylim, zlim=zlim)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    scatter = None
    colors = []

    with tqdm(total=steps) as pbar:
        def animate(step):
            global scatter, colors
            pbar.n = step
            pbar.refresh()
            coords = snapshots[:step, :, :].reshape(-1, 3).T
            if scatter is not None:
                scatter.remove()
            if step:
                colors.extend(palette)
            scatter = ax.scatter(*coords, color=colors, marker='.')
            return scatter,

        ani = FuncAnimation(fig, animate, interval=1000 / fps, frames=steps)
        ani.save('zad3.gif', dpi=dpi, writer=PillowWriter(fps=fps))
