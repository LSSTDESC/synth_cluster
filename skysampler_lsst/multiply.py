"""
Multiply two PDFs represented as MCMC chains
"""

import numpy as np
import pandas as pd
import sklearn.neighbors as neighbors
import sklearn.decomposition as decomp
import copy
import glob

