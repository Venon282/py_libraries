from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List

import matplotlib as mpl


_PALETTE_MUTED: List[str] = [
    '#1f77b4', '#d62728', '#2ca02c', '#ff7f0e',
    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
]

_PALETTE_BRIGHT: List[str] = [
    '#4477AA', '#EE6677', '#228833', '#CCBB44',
    '#66CCEE', '#AA3377', '#BBBBBB',
]

_BASE_RC: Dict[str, Any] = {
    'xtick.direction':     'in',
    'ytick.direction':     'in',
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'axes.spines.top':     False,
    'axes.spines.right':   False,
    'legend.framealpha':   0.8,
    'savefig.bbox':        'tight',
}

_PAPER_RC: Dict[str, Any] = {
    **_BASE_RC,
    'font.family':     'serif',
    'font.size':       10,
    'axes.labelsize':  10,
    'axes.titlesize':  11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi':      150,
    'savefig.dpi':     300,
    'axes.linewidth':  0.8,
    'lines.linewidth': 1.4,
    'lines.markersize': 5,
}

_TALK_RC: Dict[str, Any] = {
    **_BASE_RC,
    'font.family':     'sans-serif',
    'font.size':       14,
    'axes.labelsize':  14,
    'axes.titlesize':  16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi':      100,
    'savefig.dpi':     200,
    'axes.linewidth':  1.2,
    'lines.linewidth': 2.0,
    'lines.markersize': 8,
}

_POSTER_RC: Dict[str, Any] = {
    **_BASE_RC,
    'font.family':     'sans-serif',
    'font.size':       18,
    'axes.labelsize':  18,
    'axes.titlesize':  20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.dpi':      100,
    'savefig.dpi':     150,
    'axes.linewidth':  1.5,
    'lines.linewidth': 2.5,
    'lines.markersize': 10,
}


@dataclass
class Theme:
    name:      str
    rc_params: Dict[str, Any] = field(default_factory=dict)
    palette:   List[str]      = field(default_factory=list)

    def apply(self) -> None:
        mpl.rcParams.update(self.rc_params)
        if self.palette:
            mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=self.palette)

    def derive(self, rc_overrides: Dict[str, Any] = None,
               palette: List[str] = None) -> Theme:
        """Create a modified copy of this theme."""
        clone = copy.deepcopy(self)
        if rc_overrides:
            clone.rc_params.update(rc_overrides)
        if palette is not None:
            clone.palette = palette
        return clone


THEMES: Dict[str, Theme] = {
    'paper':   Theme('paper',   _PAPER_RC,   _PALETTE_MUTED),
    'talk':    Theme('talk',    _TALK_RC,    _PALETTE_BRIGHT),
    'poster':  Theme('poster',  _POSTER_RC,  _PALETTE_BRIGHT),
    'default': Theme('default', {},          []),
}