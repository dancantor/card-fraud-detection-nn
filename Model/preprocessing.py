import numpy as np
from typing import List

from pandas import Series


class Preprocessing:
    @staticmethod
    def standard_scale(features: Series):
        mean = features.mean()
        std_dev = features.std()
        return (features - mean) / std_dev
