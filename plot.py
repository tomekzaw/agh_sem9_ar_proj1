import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('ggplot')

metrics = {
    'speedup': 'przyspieszenia',
    'efficiency': 'efektywności',
    'serial_fraction': 'części sekwencyjnej',
}

ylabels = {
    'speedup': 'Przyspieszenie bezwzględne',
    'efficiency': 'Efektywność',
    'serial_fraction': 'Część sekwencyjna',
}


def assign_t1(df: pd.DataFrame) -> pd.DataFrame:
    df_t1 = df[df['np'] == 1].set_index('N')['time'].groupby('N').agg('min').rename('t1')
    return df.merge(df_t1, on='N')


def calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df['speedup'] = df['t1'] / df['time']
    df['efficiency'] = df['speedup'] / df['np']
    df['serial_fraction'] = (df['np'] - df['speedup']) / (df['speedup'] * (df['np'] - 1))
    return df


def group_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(by=['N', 'np'], dropna=False, as_index=False).agg({
        'speedup': ['mean', 'std'],
        'efficiency': ['mean', 'std'],
        'serial_fraction': ['mean', 'std'],
    })


def plot(df: pd.DataFrame, metric: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 6))

    grouped = df.sort_values(by=['N', 'np']).groupby('N', dropna=False, sort=False)

    colors = ['red', 'green', 'blue']

    for (problem_size, group), color in zip(grouped, colors):
        ax.errorbar(x=group['np'],
                    y=group[metric, 'mean'],
                    yerr=group[metric, 'std'],
                    label=f'rozmiar problemu: {problem_size}',
                    color=color, fmt='.', ls='dotted', capsize=4)

    xs = [df['np'].min(), df['np'].max()]
    ys = xs if metric == 'speedup' else [1, 1] if metric == 'efficiency' else None
    if ys is not None:
        ax.plot(xs, ys, ls='dashed', lw=1, color='gray', label='przebieg idealny')

    ax.set(title=f'Zależność {metrics[metric]} od liczby procesów',
           xlabel='Liczba procesów',
           ylabel=ylabels[metric],
           xticks=range(1, df['np'].max() + 1))

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    return fig


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(f'plots/{name}.png', bbox_inches='tight', dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    for zad in [1, 2, 3]:
        df = pd.read_csv(f'measurements/zad{zad}.csv')
        df = assign_t1(df)
        df = calculate_metrics(df)
        df = group_data(df)
        df['np'] = df['np'].astype(int)

        for metric in metrics:
            fig = plot(df, metric)
            save(fig, f'zad{zad}_{metric}')
