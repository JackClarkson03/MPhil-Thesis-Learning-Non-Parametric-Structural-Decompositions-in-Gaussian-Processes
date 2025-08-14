# --- Import Libraries ---

import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import sys
import time
import math
import torch
import gpflow
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from scipy.io import savemat
from scipy.io import loadmat
from gpflow.models import GPR
from gpflow.base import Parameter
from gpflow.kernels import Kernel
from gpflow.optimizers import Scipy
from gpflow.inducing_variables import InducingPoints
from ucimlrepo import fetch_ucirepo
from scipy.stats import gaussian_kde, stats, norm
from scipy.interpolate import make_interp_spline, interp1d
from collections import defaultdict
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm, Normalize
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error 
from sklearn.preprocessing import QuantileTransformer, PowerTransformer, StandardScaler



import warnings

warnings.filterwarnings(
    "ignore", 
    message="KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads.",
    category=UserWarning
)




# --- Import Github Functions ---

uci_path = os.path.abspath("examples//uci")
if uci_path not in sys.path:
    sys.path.append(uci_path)


from uci_plotting import *
from uci_regression_train import *
from uci_classification_train import *

oak_path = os.path.abspath("oak")
if oak_path not in sys.path:
    sys.path.append(oak_path)

from utils import *
from oak_kernel import *
from model_utils import *
from input_measures import *
from plotting_utils import *
from ortho_rbf_kernel import *
from normalising_flow import *
from ortho_binary_kernel import *
from ortho_categorical_kernel import *



# --- Import Custom Functions --- 
funcs_path = os.path.abspath("funcs")
if funcs_path not in sys.path:
    sys.path.append(funcs_path)
    
from helper_functions import *



# --- Set MATPLOTLIB Parameters ---
mpl.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 20,
    'axes.titlesize': 32,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 20,
    'figure.titlesize': 28,
    'axes.grid': False,
    'grid.color': 'grey',
    'grid.linestyle': '--',
    'grid.linewidth': 1.0,
})



print("All packages loaded successfully.")