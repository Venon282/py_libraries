from matplotlib import markers

class Maker:
    @staticmethod
    def all():
        """All named markers."""
        return list(markers.MarkerStyle.markers.keys())
    
    @staticmethod
    def all_not_none():
        """All named markers."""
        return list(set(markers.MarkerStyle.markers.keys()) - set(Maker.none()))

    # — filled vs. unfilled —————

    @staticmethod
    def filled():
        return list(markers.MarkerStyle.filled_markers)

    @staticmethod
    def unfilled():
        return [
            m for m in Maker.all()
            if m not in Maker.filled()
        ]

    @staticmethod
    def fillstyles():
        return list(markers.MarkerStyle.fillstyles)

    # — geometric shapes —————

    @staticmethod
    def pixel():
        return ['.', ',']

    @staticmethod
    def circle():
        return ['o']

    @staticmethod
    def square():
        return ['s']

    @staticmethod
    def diamond():
        return ['D', 'd']

    @staticmethod
    def triangle():
        return ['v', '^', '<', '>', '1', '2', '3', '4']

    @staticmethod
    def pentagon():
        return ['p', 'P']

    @staticmethod
    def hexagon():
        return ['h', 'H']

    @staticmethod
    def star():
        return ['*', 'X']

    @staticmethod
    def cross():
        return ['+', 'x']

    # — numeric / digit markers —————

    @staticmethod
    def numeric():
        return [
            m for m in Maker.all()
            if isinstance(m, int) or (isinstance(m, str) and m.isdigit())
        ]

    # — text‐style / functional —————

    @staticmethod
    def tick():
        return ['|', '_']

    @staticmethod
    def none():
        return ['None', 'none', ' ', '']

    # —组合 for directional —————

    @staticmethod
    def directional():
        return Maker.tick() + Maker.caret()

    # — everything else —————

    @staticmethod
    def other():
        known = (
            Maker.pixel()   + Maker.circle() +
            Maker.square()  + Maker.diamond() +
            Maker.triangle()+ Maker.pentagon() +
            Maker.hexagon() + Maker.star()    +
            Maker.cross()   + Maker.numeric() +
            Maker.tick()  +
            Maker.none()
        )
        return [m for m in Maker.all() if m not in known]

    # — sanity check —————

    @staticmethod
    def verify():
        recognized = (
            Maker.pixel()   + Maker.circle() +
            Maker.square()  + Maker.diamond() +
            Maker.triangle()+ Maker.pentagon() +
            Maker.hexagon() + Maker.star()    +
            Maker.cross()   + Maker.numeric() +
            Maker.tick()    +
            Maker.none()
        )
        return [m for m in recognized if m not in Maker.all()]
