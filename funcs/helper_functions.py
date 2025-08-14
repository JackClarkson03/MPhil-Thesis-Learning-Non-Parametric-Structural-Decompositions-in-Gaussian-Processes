# --- Import Libraries --- 

import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import sys
import time
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



# --- Import Github Functions --- 

from input_measures import *
from model_utils import *
from normalising_flow import *
from oak_kernel import *
from ortho_binary_kernel import *
from ortho_categorical_kernel import *
from ortho_rbf_kernel import *
from plotting_utils import *
from utils import *




def eval_component(kernel_component, alpha, X_test, X_train):
    """
    Calculates the predictive mean for a single kernel component.
    """
    K_xt = kernel_component(X_test, X_train).numpy()
    return K_xt @ alpha.numpy()


def plot_3d_surface(X, Y, Z, title, savefig=False, savestr='', xlabel="$x_1$", ylabel="$x_2$", zlabel="f", cmap="RdBu_r"):
    """
    Creates and displays a styled 3D surface plot.
    """    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor="none", alpha=0.9)

    ax.set_title(title, fontsize=40)
    ax.set_xlabel(xlabel, labelpad=10)
    ax.set_ylabel(ylabel, labelpad=10)
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.tick_params(axis='z', labelsize=10)
    plt.tight_layout()
    
    if savefig:
        plt.savefig(savestr)
    plt.show()



def get_component_variance(kernel_component, K_yy, X_test, X_train):
    """
    Computes the posterior variance for a single kernel component.
    """
    k_test_test_diag = kernel_component.K_diag(X_test)
    k_test_train = kernel_component.K(X_test, X_train)
    
    # Efficiently compute diag(k_test_train @ K_yy_inv @ k_test_train.T)
    v = tf.linalg.solve(K_yy, tf.transpose(k_test_train))
    var_reduction = tf.reduce_sum(k_test_train * tf.transpose(v), axis=1)
    
    variance = k_test_test_diag - var_reduction
    return variance
    


def test_metrics(model, X_test, y_test_standardized, y_mean, y_std):
    """
    Calculates and returns test metrics, ensuring calculations are done on the correct scale.
    """
    y_pred_standardized, _ = model.predict_f(X_test)

    # 1. NLL Calculation
    data_for_nll = (X_test, y_test_standardized.reshape(-1, 1))
    nll_model = -tf.reduce_mean(model.predict_log_density(data_for_nll))

    # 2. MSE/RMSE Calculation
    y_pred_original = (y_pred_standardized * y_std) + y_mean
    y_test_original = (y_test_standardized * y_std) + y_mean
    rmse_model = mean_squared_error(y_test_original, y_pred_original, squared=False)

    # 3. Training NMLL Calculation
    training_nmll = model.training_loss().numpy()

    return training_nmll, nll_model.numpy(), rmse_model



def compute_sobol_oak_with_mog(model: gpflow.models.BayesianModel, share_var_across_orders: bool = True):
    """
    Modified version of compute_sobol_oak that handles MOGMeasure with a single Gaussian for Sobol calculation.
    """
    assert isinstance(model.kernel, OAKKernel), "only work for OAK kernel"
    num_dims = model.data[0].shape[1]

    selected_dims_oak, kernel_list = get_list_representation(
        model.kernel, num_dims=num_dims
    )
    selected_dims_oak = selected_dims_oak[1:]  # skip constant term
    if isinstance(model, (gpflow.models.SGPR, gpflow.models.SVGP)):
        X = model.inducing_variable.Z
    else:
        X = model.data[0]
    N = X.shape[0]
    alpha = get_model_sufficient_statistics(model, get_L=False)
    sobol = []
    L_list = []
    for kernel in kernel_list:
        #assert isinstance(kernel, KernelComponenent)
        assert 'KernelComponenent' in str(type(kernel))
        if len(kernel.iComponent_list) == 0:
            continue  # skip constant term
        L = np.ones((N, N))
        n_order = len(kernel.kernels)
        for j in range(len(kernel.kernels)):
            if share_var_across_orders:
                v = kernel.oak_kernel.variances[n_order].numpy() if j < 1 else 1
            else:
                v = kernel.kernels[j].base_kernel.variance.numpy()

            dim = kernel.kernels[j].active_dims[0]

            if isinstance(kernel.kernels[j], OrthogonalRBFKernel):
                l = kernel.kernels[j].base_kernel.lengthscales.numpy()

                if isinstance(kernel.kernels[j].measure, MOGMeasure):
                    # Handle MOGMeasure by extracting mu and delta
                    if len(kernel.kernels[j].measure.means) == 1: # Assumes single-component Gaussian
                        mu = kernel.kernels[j].measure.means[0]
                        delta = np.sqrt(kernel.kernels[j].measure.variances[0])
                        L = L * compute_L(X, l, v, dim, delta, mu)
                    else:
                        raise NotImplementedError("Sobol for multi-component MOG is not implemented.")

                elif isinstance(kernel.kernels[j].measure, EmpiricalMeasure):
                     L = (
                        v ** 2
                        * L
                        * compute_L_empirical_measure(
                            kernel.kernels[j].measure.location,
                            kernel.kernels[j].measure.weights,
                            kernel.kernels[j],
                            tf.reshape(X[:, dim], [-1, 1]),
                        )
                    )
                else: # Fallback to default GaussianMeasure N(0,1)
                    mu_default = 0.0
                    delta_default = 1.0
                    L = L * compute_L(X,l,v,dim,delta_default,mu_default)


            elif isinstance(kernel.kernels[j], OrthogonalBinary):
                p0 = kernel.kernels[j].p0
                L = L * compute_L_binary_kernel(X, p0, v, dim)

            elif isinstance(kernel.kernels[j], OrthogonalCategorical):
                p = kernel.kernels[j].p
                W = kernel.kernels[j].W
                kappa = kernel.kernels[j].kappa
                L = L * compute_L_categorical_kernel(X, W, kappa, p, v, dim)

            else:
                raise NotImplementedError
        L_list.append(L)
        mean_term = tf.tensordot(
            tf.tensordot(tf.transpose(alpha), L, axes=1), alpha, axes=1
        ).numpy()[0][0]
        sobol.append(mean_term)

    assert len(selected_dims_oak) == len(sobol)
    return selected_dims_oak, sobol
