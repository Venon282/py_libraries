from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class AxisConfig:
    # Labels
    title:  str = ''
    xlabel: str = ''
    ylabel: str = ''

    # Limits and scale
    xlim:   Optional[Tuple[float, float]] = None
    ylim:   Optional[Tuple[float, float]] = None
    xscale: Optional[str]                 = None
    yscale: Optional[str]                 = None

    # Grid
    grid: Union[bool, Dict[str, Any]] = True

    # Legend
    legend:      bool           = False
    legend_opts: Dict[str, Any] = field(default_factory=dict)

    # Ticks
    xticks:      Optional[List[float]] = None
    yticks:      Optional[List[float]] = None
    xticklabels: Optional[List[str]]   = None
    yticklabels: Optional[List[str]]   = None

    @classmethod
    def fieldNames(cls) -> frozenset:
        return frozenset(f.name for f in fields(cls))

    @classmethod
    def fromKwargs(cls, kwargs: dict) -> AxisConfig:
        """Pop all recognized AxisConfig fields from kwargs and return an instance."""
        known = cls.fieldNames()
        init_args = {k: kwargs.pop(k) for k in list(kwargs) if k in known}
        return cls(**init_args)