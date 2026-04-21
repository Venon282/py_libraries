import numpy as np
from .Figure import Figure


def unicAxe():
    t  = np.linspace(0, 2 * np.pi, 300)
    y1 = np.sin(t)
    y2 = np.cos(t) * 0.6

    with Figure(theme='paper', figsize=(8, 4)) as fig:
        ax = fig.addAxes(
            title='Trigonometric signals',
            xlabel='t (rad)',
            ylabel='Amplitude',
            legend=True,
        )
        ax.plot(
            (t, y1, {'label': 'sin(t)'}),
            (t, y2, {'label': '0.6 cos(t)', 'linestyle': '--'}),
        )
        fig.save('out/unic_axe.pdf')


def chaining():
    t     = np.linspace(0, 2 * np.pi, 300)
    y1    = np.sin(t)
    t_obs = t[::15]
    y_obs = np.sin(t_obs) + np.random.normal(0, 0.08, len(t_obs))

    fig = Figure(theme='paper')
    ax  = fig.addAxes()

    ax.plot((t, y1, {'label': 'fit'})) \
      .scatter((t_obs, y_obs, {'s': 12, 'label': 'data'})) \
      .style(
          title='Fit vs. observations',
          xlabel='t', ylabel='y',
          legend=True, ylim=(-1.5, 1.5),
      )

    fig.show()


def gridAndSubGraphs():
    rng    = np.random.default_rng(42)
    data_a = rng.lognormal(mean=0.0, sigma=0.6, size=2000)
    data_b = rng.normal(loc=0.0,    scale=1.0,  size=2000)
    x      = rng.normal(0, 1, 500)
    y      = x * 0.8 + rng.normal(0, 0.4, 500)
    px     = rng.normal(0, 1, 3000)
    py     = px * 0.5 + rng.normal(0, 0.5, 3000)

    fig  = Figure(theme='paper', figsize=(12, 8))
    grid = fig.addSubplots(2, 2)

    grid[0][0].histogram(data_a, bins=50, bin_type='log') \
              .style(title='Distribution A', xlabel='value', xscale='log')

    grid[0][1].scatterDensity((x, y), cbar_label='density') \
              .style(title='2D density')

    grid[1][0].heatmap((px, py), bins=80, cbar_label='counts') \
              .style(title='Heatmap')

    grid[1][1].ecdf(data_b) \
              .style(title='ECDF', xlabel='value', ylabel='F(x)')

    fig.save('out/panel.png')


def twinAxes():
    t    = np.linspace(0, 10, 300)
    temp = 300 + 50 * np.sin(t)
    pres = 1e5 + 2e4 * np.cos(t * 1.3)

    fig = Figure(theme='talk', figsize=(9, 5))
    ax1 = fig.addAxes(xlabel='Time (s)', ylabel='Temperature (K)')
    ax2 = fig.addTwin(ax1, axis='x', ylabel='Pressure (Pa)')

    ax1.plot((t, temp, {'color': '#1f77b4', 'label': 'T'}))
    ax2.plot((t, pres, {'color': '#d62728', 'label': 'P'}))

    fig.show()


def insetZoom():
    t      = np.linspace(0, 4 * np.pi, 600)
    signal = np.sin(t) * np.exp(-t * 0.1)

    mask        = (t >= 1.2) & (t <= 1.8)
    t_zoom      = t[mask]
    signal_zoom = signal[mask]

    fig = Figure(theme='paper')
    ax  = fig.addAxes(title='Full signal + zoom', xlabel='t', ylabel='y')
    ins = fig.addInset(ax, bounds=(0.55, 0.50, 0.40, 0.35))

    ax.plot((t, signal))
    ins.plot((t_zoom, signal_zoom)) \
       .style(xlim=(1.2, 1.8), grid=False)

    fig.save('out/zoom.pdf')


def rawAccess():
    t  = np.linspace(0, 2 * np.pi, 300)
    y1 = np.sin(t)

    fig = Figure(theme='paper')
    ax  = fig.addAxes(title='Custom annotation', xlabel='t', ylabel='y')
    ax.plot((t, y1))

    ax.raw.annotate(
        'peak',
        xy=(np.pi / 2, 1.0), xytext=(2.5, 0.75),
        arrowprops=dict(arrowstyle='->', color='gray'),
        fontsize=9,
    )
    fig.raw.suptitle('Supplementary figure S1', fontsize=9)

    fig.show()


if __name__ == '__main__':
    unicAxe()
    chaining()
    gridAndSubGraphs()
    twinAxes()
    insetZoom()
    rawAccess()