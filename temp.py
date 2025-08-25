import datetime
import time
import numpy as np
from .number import randomGaussianWithBorn

def scheduleTomorrowExecution(
    period,
    distribution: str = "gaussian",
    possibilities: dict | None = None,
    verbose: int = 1
):
    """
    Schedule execution for tomorrow within a specified time period.

    Parameters
    ----------
    period : str | list[float]
        - If str: must be a key in `possibilities` (e.g., "morning", "night").
        - If list: must contain exactly two floats/ints [hour_start, hour_end].
    distribution : str, default="gaussian"
        - "linear" : uniform random hour between start and end.
        - "gaussian"/"normal" : Gaussian random hour within range (via getRandomGaussianWithBorn).
    possibilities : dict, optional
        Mapping of named periods to [hour_start, hour_end].
        If None, defaults to:
            {
                "morning"   : [6, 12],
                "noon"      : [12, 14],
                "afternoon" : [12, 18],
                "evening"   : [18, 23],
                "night"     : [1, 6],
                "day"       : [6, 18],
                "all"       : [6, 23],
            }
    verbose : int, default=1
        If >0, prints the chosen execution time.

    Raises
    ------
    ValueError
        If `period` is invalid or `distribution` is unsupported.
    TypeError
        If `period` is not str or list.

    Behavior
    --------
    - Chooses a random hour in the requested period (linear or gaussian).
    - Schedules execution time for *tomorrow*.
    - Sleeps until that time is reached.
    """

    if possibilities is None:
        possibilities = {
            "morning"   : [6, 12],
            "noon"      : [12, 14],
            "afternoon" : [12, 18],
            "evening"   : [18, 23],
            "night"     : [1, 6],
            "day"       : [6, 18],
            "all"       : [6, 23],
        }

    # Validate and resolve period
    if isinstance(period, str):
        if period not in possibilities:
            raise ValueError(
                f"Invalid time period '{period}'. "
                f"Valid options: {', '.join(possibilities)}."
            )
        period = possibilities[period]
    elif isinstance(period, list):
        if len(period) != 2:
            raise ValueError(
                "When period is a list, it must contain exactly 2 values [hour1, hour2]."
            )
    else:
        raise TypeError(f"Period of type {type(period)} is not supported.")

    now = datetime.datetime.now()
    tomorrow = now + datetime.timedelta(days=1)

    # Pick random hour
    if distribution == "linear":
        target_hour = np.random.uniform(*period)
    elif distribution in ("gaussian", "normal"):
        target_hour = randomGaussianWithBorn(*period)
    else:
        raise ValueError("Distribution must be 'linear' or 'gaussian'/'normal'.")

    # Convert fractional hour into hh:mm:ss
    hour = int(target_hour)
    minute = int((target_hour - hour) * 60)
    second = int((((target_hour - hour) * 60) - minute) * 60)

    target_time = datetime.datetime(
        year=tomorrow.year,
        month=tomorrow.month,
        day=tomorrow.day,
        hour=hour,
        minute=minute,
        second=second,
    )

    sleep_seconds = (target_time - now).total_seconds()

    if verbose > 0:
        print("Next execution time:", target_time.strftime("%d/%m/%Y %H:%M:%S"))

    time.sleep(sleep_seconds)