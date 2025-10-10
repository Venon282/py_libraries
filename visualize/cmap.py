import matplotlib.pyplot as plt

class Cmap:
    @staticmethod
    def all():
        """All named colorscales (with both normal and reversed names)."""
        return list(plt.colormaps())

    # Sequential (single-hue or multi-hue)
    @staticmethod
    def sequential():
        return [
            'magma', 'inferno', 'plasma', 'viridis', 'cividis',
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'binary', 'BuGn', 'BuPu', 'GnBu', 'OrRd',
            'PuBu', 'PuBuGn', 'PuRd', 'RdPu', 'YlGn', 'YlGnBu',
            # additional sequential/miscellaneous
            # 'afmhot', 'autumn', 'bone', 'brg', 'cool', 'copper',
            # 'cubehelix', 'flag', 'gist_earth', 'gist_gray', 'gist_heat', 'gist_ncar',
            # 'gnuplot', 'gnuplot2', 'gray', 'hot', 'jet', 'nipy_spectral',
            # 'ocean', 'pink', 'prism', 'rainbow', 'summer', 'terrain', 'winter',
            # 'turbo', 'spring', 'seismic', 'gist_rainbow', 'gist_stern', 'gist_yarg',
            # 'grey', 'gist_grey', 'gist_yerg', 'Grays', 'rocket'
        ]

    @staticmethod
    def sequential_r():
        all_ = Cmap.all()
        return [m + '_r' for m in Cmap.sequential() if m + '_r' in all_]

    # Diverging
    @staticmethod
    def diverging():
        return [
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy',
            'RdYlBu', 'RdBu', 'RdYlGn', 'Spectral',
            'coolwarm', 'bwr'
        ]

    @staticmethod
    def diverging_r():
        all_ = Cmap.all()
        return [m + '_r' for m in Cmap.diverging() if m + '_r' in all_]

    # Cyclical
    @staticmethod
    def cyclical():
        return [
            'twilight', 'twilight_shifted', 'hsv',
            # additional cyclic
            'flag', 'prism'
        ]

    @staticmethod
    def cyclical_r():
        all_ = Cmap.all()
        return [m + '_r' for m in Cmap.cyclical() if m + '_r' in all_]

    # Qualitative (discrete/category)
    @staticmethod
    def qualitative():
        return [
            'tab10', 'tab20', 'tab20b', 'tab20c',
            'Pastel1', 'Pastel2', 'Paired', 'Accent',
            'Dark2', 'Set1', 'Set2', 'Set3',
            # others
            'Wistia', 'CMRmap', 'flare', 'crest', 'icefire', 'mako', 'vlag'
        ]

    @staticmethod
    def qualitative_r():
        all_ = Cmap.all()
        return [m + '_r' for m in Cmap.qualitative() if m + '_r' in all_]

    @staticmethod
    def other():
        all_ = Cmap.all()
        known = (
            Cmap.sequential() + Cmap.sequential_r() +
            Cmap.diverging() + Cmap.diverging_r() +
            Cmap.cyclical() + Cmap.cyclical_r() +
            Cmap.qualitative() + Cmap.qualitative_r()
        )
        return [a for a in all_ if a not in known]

    @staticmethod
    def verify():
        all_ = Cmap.all()
        recognized = (
            Cmap.sequential() + Cmap.sequential_r() +
            Cmap.diverging() + Cmap.diverging_r() +
            Cmap.cyclical() + Cmap.cyclical_r() +
            Cmap.qualitative() + Cmap.qualitative_r()
        )
        return [r for r in recognized if r not in all_]