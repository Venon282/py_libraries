import random
import colorsys
from matplotlib import colors

class Color:
    @staticmethod
    def all():
        """All named colors from matplotlib (base, CSS4, Tableau, XKCD)."""
        return list(colors.get_named_colors_mapping().keys())

    @staticmethod
    def base():
        """Base colors (short names)."""
        return list(colors.BASE_COLORS.keys())

    @staticmethod
    def tableau():
        """Tableau colors (named palette colors)."""
        return list(colors.TABLEAU_COLORS.keys())

    @staticmethod
    def css4():
        """CSS4 named colors."""
        return list(colors.CSS4_COLORS.keys())

    @staticmethod
    def xkcd():
        """XKCD survey colors (prefixed names)."""
        return list(colors.XKCD_COLORS.keys())

    @staticmethod
    def hex_values():
        """All hex values of named colors."""
        return list(colors.get_named_colors_mapping().values())

    @staticmethod
    def other():
        """Colors not in base, tableau, CSS4, or XKCD."""
        all_ = set(Color.all())
        known = set(Color.base() + Color.tableau() + Color.css4() + Color.xkcd())
        return list(all_ - known)

    @staticmethod
    def luminance(color_name):
        """Return relative luminance [0..1] of a named color."""
        rgb = colors.to_rgb(colors.get_named_colors_mapping()[color_name])
        # Standard Rec. 709 luminance
        return 0.2126*rgb[0] + 0.7152*rgb[1] + 0.0722*rgb[2]

    @staticmethod
    def light():
        """Named colors whose names start with 'light' (case-insensitive)."""
        return [name for name in Color.all() if name.lower().startswith('light')]

    @staticmethod
    def dark():
        """Named colors whose names start with 'dark' (case-insensitive)."""
        return [name for name in Color.all() if name.lower().startswith('dark')]

    @staticmethod
    def dark(threshold=0.5):
        """Named colors with luminance at or below threshold (dark colors)."""
        return [name for name in Color.all() if Color.luminance(name) <= threshold]

    @staticmethod
    def grayscale(tolerance=1e-6):
        """Named colors where R≈G≈B within tolerance."""
        result = []
        for name, hexcode in colors.get_named_colors_mapping().items():
            r, g, b = colors.to_rgb(hexcode)
            if abs(r-g) < tolerance and abs(g-b) < tolerance:
                result.append(name)
        return result

    @staticmethod
    def primary(dominance=0.1):
        """Colors where one channel exceeds the other two by 'dominance'."""
        prim = {'red': [], 'green': [], 'blue': []}
        for name, hexcode in colors.get_named_colors_mapping().items():
            r, g, b = colors.to_rgb(hexcode)
            if r - max(g, b) > dominance:
                prim['red'].append(name)
            if g - max(r, b) > dominance:
                prim['green'].append(name)
            if b - max(r, g) > dominance:
                prim['blue'].append(name)
        return prim

    @staticmethod
    def complementary(color_name):
        """Return hex code for the complementary color of a named color."""
        hexcode = colors.get_named_colors_mapping()[color_name]
        r, g, b = colors.to_rgb(hexcode)
        comp = (1-r, 1-g, 1-b)
        return colors.to_hex(comp)

    @staticmethod
    def lighten(color_name, factor=0.2):
        """Lighten a named color by a given factor (0..1)."""
        hexcode = colors.get_named_colors_mapping()[color_name]
        r, g, b = colors.to_rgb(hexcode)
        r, g, b = [min(1, c + factor*(1-c)) for c in (r, g, b)]
        return colors.to_hex((r, g, b))

    @staticmethod
    def darken(color_name, factor=0.2):
        """Darken a named color by a given factor (0..1)."""
        hexcode = colors.get_named_colors_mapping()[color_name]
        r, g, b = colors.to_rgb(hexcode)
        r, g, b = [max(0, c*(1-factor)) for c in (r, g, b)]
        return colors.to_hex((r, g, b))

    @staticmethod
    def sorted_by_hue(include_hex=False):
        """Return colors sorted by HSV hue. Optionally include hex codes."""
        items = []
        for name, hexcode in colors.get_named_colors_mapping().items():
            r, g, b = colors.to_rgb(hexcode)
            h, l, s = colorsys.rgb_to_hls(r, g, b)
            items.append((h, name, hexcode))
        items.sort(key=lambda x: x[0])
        return [(name, hex) if include_hex else name for _, name, hex in items]

    @staticmethod
    def random(n=5, seed=None):
        """Random subset of n named colors."""
        names = Color.all()
        if seed is not None:
            random.seed(seed)
        return random.sample(names, min(n, len(names)))

    @staticmethod
    def verify():
        """Verify that named mappings are consistent and unique."""
        mapping = colors.get_named_colors_mapping()
        names = list(mapping.keys())
        unique = len(names) == len(set(names))
        duplicates = [name for name in set(names) if names.count(name) > 1]
        return {
            'total': len(names),
            'unique': unique,
            'duplicates': duplicates
        }