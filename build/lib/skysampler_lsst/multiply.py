"""
Multiply two PDFs represented as a set of samples or MCMC chains
"""

import numpy as np
import pandas as pd
import sklearn.neighbors as neighbors
import sklearn.decomposition as decomp
import copy
import glob
import multiprocessing as mp

from .utils import partition


