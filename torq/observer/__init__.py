from .base import BaseObserver
from .ema import EmaObserver
from .minmax import MinmaxObserver
from .percentile import PercentileObserver

OBSERVER_DICT = {
    'ema': EmaObserver,
    'minmax': MinmaxObserver,
    'percentile': PercentileObserver
}

