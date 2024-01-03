#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 17:27:57 2023

@author: edoardochidichimo

METRICS:
    
    UNIVARIATE & JOINT ENTROPY
    MUTUAL INFORMATION
    SYMMETRIC UNCERTAINTY
    TRANSFER ENTROPY (C) / CONDITIONAL MUTUAL INFORMATION (D)
    GRANGER EMERGENCE


APPROACHES (Discrete [D] and Continuous [C] Estimators):

    [D] BINNING
    [D] WEIGHTED SYMBOLIC 
    [C] GAUSSIAN COPULAS 
    [C] KERNEL-BASED
    [C] k-NEAREST NEIGHBOUR
    

"""

# CORE
import io
from pathlib import Path
from copy import copy
from collections import OrderedDict

# DATA SCIENCE
import numpy as np
import scipy as sp
from scipy import stats
from scipy.integrate import odeint
from collections import Counter

# VISUALISATION
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import ScalarFormatter
from matplotlib.colors import Normalize
from matplotlib.cm import viridis
from mpl_toolkits.axes_grid1 import make_axes_locatable

from hypyp.ext.mpl3d import glm
from hypyp.ext.mpl3d.mesh import Mesh
from hypyp.ext.mpl3d.camera import Camera

# MNE
import mne
from mne.channels import make_standard_montage
from mne.viz import plot_topomap




# =============================================================================
# BINNING: ENTROPY, MUTUAL INFORMATION, CONDITIONAL MUTUAL INFORMATION
# =============================================================================


def calc_pmd(X: np.array, bins: int):
    
    edges_X = np.linspace(min(X), max(X), bins + 1)
    X_which_bin = np.digitize(X, edges_X, right=False)
    counts_X = np.bincount(X_which_bin, minlength=bins + 1)[1:]
    pmd_X = counts_X.astype(float) / len(X)
    
    return pmd_X

def test_calc_jpmd(X: np.array, Y:np.array, bins: int, Z=None):
    edges_X, edges_Y = np.linspace(min(X), max(X), bins + 1), np.linspace(min(Y), max(Y), bins + 1)
    X_which_bin, Y_which_bin = np.digitize(X, edges_X, right=False), np.digitize(Y, edges_Y, right=False)
    mask_X, mask_Y = (X_which_bin[:, None] == np.arange(1, bins + 2)).astype(int), (Y_which_bin[:, None] == np.arange(1, bins + 2)).astype(int)

    if Z is not None:
        
        assert Z.shape == X.shape == Y.shape, "All arrays should have the same shape"
        
        edges_Z = np.linspace(min(Z), max(Z), bins + 1)
        Z_which_bin = np.digitize(Z, edges_Z, right=False)
        
        comb_freq = np.zeros((bins + 1, bins + 1, bins + 1))

        for i in range(bins + 1):
            for j in range(bins + 1):
                for k in range(bins + 1):
                    XYZ_in_bin = (X_which_bin == i + 1) & (Y_which_bin == j + 1) & (Z_which_bin == k + 1)
                    comb_freq[i, j, k] = np.sum(XYZ_in_bin)

        comb_pmf = comb_freq / np.sum(comb_freq)
        return np.array(comb_pmf)

    else:
        joint_pmd = np.dot(mask_X.T, mask_Y).astype(float)
        return joint_pmd / np.sum(joint_pmd)

def calc_jpmd(X: np.array, Y:np.array, bins: int, Z=None):
    
    edges_X = np.linspace(min(X), max(X), bins + 1)
    edges_Y = np.linspace(min(Y), max(Y), bins + 1)
    X_which_bin = np.digitize(X, edges_X, right=False)
    Y_which_bin = np.digitize(Y, edges_Y, right=False)
    
    mask_X = (X_which_bin[:, None] == np.arange(1, bins + 2)).astype(int)
    mask_Y = (Y_which_bin[:, None] == np.arange(1, bins + 2)).astype(int)
    
    if Z is not None:
        
        assert Z.shape == X.shape == Y.shape, "All arrays should have the same shape"
        
        edges_Z = np.linspace(min(Z), max(Z), bins + 1)
        Z_which_bin = np.digitize(Z, edges_Z, right=False)
        
        comb_freq = np.zeros((bins+1, bins+1, bins+1))
        
        for i in range(bins+1):
            
            X_in_bin = (X_which_bin == i+1).astype(int)
            
            for j in range(bins+1):
                
                Y_in_bin = (Y_which_bin == j+1).astype(int)
                
                for k in range(bins+1):
                    
                    Z_in_bin = (Z_which_bin == k+1).astype(int)
                    
                    XYZ_in_bin = np.logical_and(np.logical_and(X_in_bin, Y_in_bin), Z_in_bin).astype(int)
                    
                    tot_joint = np.sum(XYZ_in_bin)
                    comb_freq[i,j, k] = tot_joint

        comb_pmf = comb_freq / np.sum(comb_freq)
        
        #mask_Z = (Z_which_bin[:, None] == np.arange(1, bins + 2)).astype(int)
        
        return np.array(comb_pmf)
        

    else:
        joint_pmd = np.dot(mask_X.T, mask_Y).astype(float)
        joint_pmd /= np.sum(joint_pmd)
        return joint_pmd


def entropy_hist(epo1: mne.Epochs, epo2: mne.Epochs, vis = False):
    
    '''
    Parameters
    ----------
    epo1 : mne.EpochsFIF
        Signal 1.
    epo2 : mne.EpochsFIF
        Signal 2.
    vis : TYPE, optional
        Whether to produce plots of average Shannon entropies. 
        The default is False.

    Returns
    -------
    np.array([H(X), H(Y), H(X,Y)]) 
        Average Shannon entropy *per channel*, both univariate and joint, 
        using histogram/binning method!
        3 columns, n_epo rows.

    '''
    
    a = epo1.get_data(copy=False)
    b = epo2.get_data(copy=False)

    assert a.shape == b.shape, "Both signals must have the same shape"
    
    n_epo, n_ch, n_samples = a.shape
    avg_ch_entropies = []
    
    for ch_i in range(n_ch):
        
        avg_ch_entropies_X = 0
        avg_ch_entropies_Y = 0
        avg_ch_jpmd = 0
        
        for epo_j in range(n_epo):
        
            Xi, Yi = a[epo_j, ch_i, :], b[epo_j, ch_i, :]

            # Freedman-Diaconis Rule for Frequency-Distribution Bin Size
            fd_bins_X = np.ceil(np.ptp(Xi) / (2.0 * stats.iqr(Xi) * len(Xi)**(-1/3)))
            fd_bins_Y = np.ceil(np.ptp(Yi) / (2.0 * stats.iqr(Yi) * len(Yi)**(-1/3)))
            fd_bins = int(np.ceil((fd_bins_X+fd_bins_Y)/2))
    
            # Calculate Univariate and Joint Probability Mass Distributions
            pmd_X = calc_pmd(Xi, fd_bins)
            pmd_Y = calc_pmd(Yi, fd_bins)
            jpmd = calc_jpmd(Xi, Yi, fd_bins)

            # Calculate Univariate Shannon Entropy
            avg_ch_entropies_X += -np.sum(pmd_X * np.log2(pmd_X + np.finfo(float).eps))
            avg_ch_entropies_Y += -np.sum(pmd_Y * np.log2(pmd_Y + np.finfo(float).eps))
            avg_ch_jpmd += -np.sum(jpmd * np.log2(jpmd + np.finfo(float).eps))
            
        avg_ch_entropies_X /= n_epo
        avg_ch_entropies_Y /= n_epo
        avg_ch_jpmd /= n_epo
        
        avg_ch_entropies.append([avg_ch_entropies_X, avg_ch_entropies_Y, avg_ch_jpmd])
    
    return np.array(avg_ch_entropies)

def mi_hist(epo1: mne.Epochs, epo2: mne.Epochs, vis = False):
    
    entropies = entropy_hist(epo1, epo2)
    MI = entropies[:,0] + entropies[:,1] - entropies[:,2]
    
    if vis:
        entropy_mi = np.column_stack((entropies[:, 0], entropies[:, 1], MI))
        plot_entropy_mi(epo1, epo2, entropy_mi)
    
    return MI

def cmi_hist(epo1: mne.Epochs, epo2: mne.Epochs, l = 3, m = 3):
    
    # This is based on https://github.com/tysonpond/symbolic-transfer-entropy/blob/master/symbolic_TE.py
    # ALSO CHECK https://github.com/mariogutierrezroig/smite/blob/master/smite/core.py
    
    a = epo1.get_data(copy=False)
    b = epo2.get_data(copy=False)

    assert a.shape == b.shape, "Both signals must have the same shape"
    
    n_epo, n_ch, _ = a.shape
    #n_samples = a.shape[2] - l  # when l=3, n_samples = 501-3 = 498
    
    Xi = a[0][0][:-l]
    Yi = b[0][0][:-l]
    Zi = b[0][0][l:]
    
    fd_bins_X = np.ceil(np.ptp(Xi) / (2.0 * stats.iqr(Xi) * len(Xi)**(-1/3)))
    fd_bins_Y = np.ceil(np.ptp(Yi) / (2.0 * stats.iqr(Yi) * len(Yi)**(-1/3)))
    fd_bins_Z = np.ceil(np.ptp(Zi) / (2.0 * stats.iqr(Zi) * len(Zi)**(-1/3)))
    fd_bins = int(np.ceil((fd_bins_X+fd_bins_Y+fd_bins_Z)/3))
    
    #pmd_X = calc_pmd(Xi, fd_bins)
    #pmd_Y = calc_pmd(Yi, fd_bins)
    pmd_Z = calc_pmd(Zi, bins=fd_bins)
    #pmd_XY = calc_jpmd(Xi, Yi, fd_bins)
    pmd_XZ = test_calc_jpmd(Xi, Zi, fd_bins)
    pmd_YZ = test_calc_jpmd(Yi, Zi, fd_bins)
    pmd_XYZ = test_calc_jpmd(Xi, Yi, fd_bins, Z=Zi)
    
    pmd_X_given_Z = pmd_XZ / (pmd_Z + np.finfo(float).eps)
    pmd_X_given_YZ = pmd_XYZ / (pmd_YZ + np.finfo(float).eps)
    
    
    
    cmi_1 = -np.sum(pmd_XYZ * np.log2((pmd_X_given_YZ / (pmd_X_given_Z + np.finfo(float).eps)) + np.finfo(float).eps))
    cmi_2 = -np.sum(pmd_XZ * np.log2(pmd_X_given_Z + np.finfo(float).eps) + pmd_XYZ * np.log2(pmd_X_given_YZ + np.finfo(float).eps))
    
    print(cmi_1)
    print(cmi_2)
    
    # P(A|B) = P(A;B) / P(B)
    
    #          p(XYZ) * log(P(X|YZ) / P(X|Z))
    # I(X;Y) = H(H|Z) - H(X|Y,Z) 
    #        = −∑x,z P(x,z)logP(x∣z) + ∑x,y,z P(x,y,z)logP(x∣y,z)
    
    return None

#CMI NEEDS CHECKING (which jpmd is correct?)
#OTHER TE METHODS?


# =============================================================================
# SYMBOLIC ENTROPY, MUTUAL INFORMATION, TRANSFER ENTRTOPY
# =============================================================================

def symbolise(X: np.ndarray, l=3, m=3):
    
    Y = np.empty((m, len(X) - (m - 1) * l))
    for i in range(m):
        Y[i] = X[i * l:i * l + Y.shape[1]]
    return Y.T
        
def incr_counts(key,d):
    d[key] = d.get(key, 0) + 1

def normalise(d):
    s=sum(d.values())        
    for key in d:
        d[key] /= s


def entropy_symb(epo1: mne.Epochs, epo2: mne.Epochs, l=3, m=3):
    
    a = epo1.get_data(copy=False)
    b = epo2.get_data(copy=False)
    
    n_epo, n_ch, n_samples = a.shape
    avg_ch_entropies = []
    
    hashmult = np.power(m, np.arange(m))
        
    
    for ch_i in range(n_ch):
        
        avg_ch_entropies_X = 0
        avg_ch_entropies_Y = 0
        avg_ch_entropies_XY = 0
    
        for epo_j in range(n_epo):
        
            X, Y = a[epo_j, ch_i, :], b[epo_j, ch_i, :]
    
            X = symbolise(X, l, m).argsort(kind='quicksort')
            Y = symbolise(Y, l, m).argsort(kind='quicksort')
    
            hashval_X = (np.multiply(X, hashmult)).sum(1) # multiply each symbol [1,0,3] by hashmult [1,3,9] => [1,0,27] and give a final array of the sum of each code ([.., .., 28, .. ])
            hashval_Y = (np.multiply(Y, hashmult)).sum(1)
            
            x_sym_to_perm = hashval_X
            y_sym_to_perm = hashval_Y
            
            p_xy = {}
            p_x = {}
            p_y = {}
            
            for i in range(len(x_sym_to_perm)-1):
                xy = str(x_sym_to_perm[i]) + "," + str(y_sym_to_perm[i])
                x = str(x_sym_to_perm[i])
                y = str(y_sym_to_perm[i])
                
                incr_counts(xy,p_xy)
                incr_counts(x,p_x)
                incr_counts(y,p_y)
                
            normalise(p_xy)
            normalise(p_x)
            normalise(p_y)
            
            p_xy = np.array(list(p_xy.values()))
            p_x = np.array(list(p_x.values()))
            p_y = np.array(list(p_y.values()))
            
            avg_ch_entropies_XY += -np.sum(p_xy * np.log2(p_xy + np.finfo(float).eps))
            avg_ch_entropies_X += -np.sum(p_x * np.log2(p_x + np.finfo(float).eps))
            avg_ch_entropies_Y += -np.sum(p_y * np.log2(p_y + np.finfo(float).eps))
    
        avg_ch_entropies_X /= n_epo
        avg_ch_entropies_Y /= n_epo
        avg_ch_entropies_XY /= n_epo
        
        avg_ch_entropies.append([avg_ch_entropies_X, avg_ch_entropies_Y, avg_ch_entropies_XY])
    
    return np.array(avg_ch_entropies)

def mi_symb(epo1: mne.Epochs, epo2: mne.Epochs, l=3, m=3, vis=True):
    
    entropies = entropy_symb(epo1, epo2)
    MI = entropies[:,0] + entropies[:,1] - entropies[:,2]
    
    if vis:
        entropy_mi = np.column_stack((entropies[:, 0], entropies[:, 1], MI))
        plot_entropy_mi(epo1, epo2, entropy_mi)
    
    return MI

def te_symb(epo1: mne.Epochs, epo2: mne.Epochs, l=3, m=3):

    
    X = symbolise(epo1.get_data(copy=False)[0][0][:], l, m).argsort(kind='quicksort')
    Y = symbolise(epo2.get_data(copy=False)[0][0][:], l, m).argsort(kind='quicksort')    
    
    hashmult = np.power(m, np.arange(m))
    hashval_X = (np.multiply(X, hashmult)).sum(1) # multiply each symbol [1,0,3] by hashmult [1,3,9] => [1,0,27] and give a final array of the sum of each code ([.., .., 28, .. ])
    hashval_Y = (np.multiply(Y, hashmult)).sum(1)
    
    x_sym_to_perm = hashval_X
    y_sym_to_perm = hashval_Y #len = 495
    
    p_xyz = {}
    p_xy = {}
    p_yz = {}
    p_y = {}
    
    for i in range(len(y_sym_to_perm)-1):
        xyz = str(x_sym_to_perm[i]) + "," + str(y_sym_to_perm[i]) + "," + str(y_sym_to_perm[i+1])
        xy = str(x_sym_to_perm[i]) + "," + str(y_sym_to_perm[i])
        yz = str(y_sym_to_perm[i]) + "," + str(y_sym_to_perm[i+1])
        y = str(y_sym_to_perm[i])
        #z = str(z_sym_to_perm[i])

        incr_counts(xyz,p_xyz)
        incr_counts(xy,p_xy)
        incr_counts(yz,p_yz)
        incr_counts(y,p_y)
        
    normalise(p_xyz)
    normalise(p_xy)
    normalise(p_yz)
    normalise(p_y)
    
    p_z_given_xy = p_xyz.copy()
    for key in p_z_given_xy:
        xy_symb = key.split(",")[0] + "," + key.split(",")[1]
        p_z_given_xy[key] /= p_xy[xy_symb]    
    # also works: p_z_given_xy = {xyz: p_xyz[xyz] / p_xy[xyz.split(",")[0] + "," + xyz.split(",")[1]] for xyz in p_xyz}  
    
    p_z_given_y = p_yz.copy()
    for key in p_z_given_y:
        y_symb = key.split(",")[0]
        p_z_given_y[key] /= p_y[y_symb]
        
    
    # p_x_given_yz = p_xyz.copy()
    # for key in p_x_given_yz:
    #     yz_symb = key.split(",")[1] + "," + key.split(",")[2]
    #     p_x_given_yz[key] /= p_yz[yz_symb]
        
    # p_x_given_z = 

    final_sum = 0
    for key in p_xyz:
        yz_symb = key.split(",")[1] + "," +  key.split(",")[2] 
        if key in p_z_given_xy and yz_symb in p_z_given_y:
            if float('-inf') < float(np.log2(p_z_given_xy[key]/p_z_given_y[yz_symb])) < float('inf'):
                final_sum += p_xyz[key]*np.log2(p_z_given_xy[key]/p_z_given_y[yz_symb])
    
    return final_sum



# =============================================================================
# GAUSSIAN COPULA ENTROPY, MUTUAL INFORMATION, 
# =============================================================================

# Gaussian Copula has been copied verbatim from Ince!

def cop_transform(X: np.array):
    
    Xi = np.argsort(np.atleast_2d(X))
    Xr = np.argsort(Xi)
    cX = (Xr+1).astype(np.float) / (Xr.shape[-1]+1)
    return cX

def copnorm(X: np.array):
    
    # Compute Empirical CDF
    xi = np.argsort(np.atleast_2d(X))
    xr = np.argsort(xi) # Why ARGsort twice? (Argsort finds INDICES whilst sort actually takes the values of the array and moves them)
    ecdf = (xr+1).astype(float) / (xr.shape[-1]+1)
    
    # Compute the inverse CDF (aka percent-point function or quantile function or inverse normal distribution)
    cx = sp.special.ndtri(ecdf) #around xbar = 0, sd = 1 (since it is normal!)
    return cx
    
def ent_g(x, biascorrect=True):
    """Entropy of a Gaussian variable in bits

    H = ent_g(x) returns the entropy of a (possibly 
    multidimensional) Gaussian variable x with bias correction.
    Columns of x correspond to samples, rows to dimensions/variables. 
    (Samples last axis)
    
    BASED ON INCE'S GCMI TOOLBOX

    """
    x = np.atleast_2d(x)
    if x.ndim > 2:
        raise ValueError("x must be at most 2d")
    Ntrl = x.shape[1]
    Nvarx = x.shape[0]

    # demean data
    x = x - x.mean(axis=1)[:,np.newaxis]
    # covariance
    C = np.dot(x,x.T) / float(Ntrl - 1)
    chC = np.linalg.cholesky(C)

    # entropy in nats
    HX = np.sum(np.log(np.diagonal(chC))) + 0.5*Nvarx*(np.log(2*np.pi)+1.0)

    ln2 = np.log(2)
    if biascorrect:
        psiterms = sp.special.psi((Ntrl - np.arange(1,Nvarx+1).astype(float))/2.0) / 2.0
        dterm = (ln2 - np.log(Ntrl-1.0)) / 2.0
        HX = HX - Nvarx*dterm - psiterms.sum()

    # convert to bits
    return HX / ln2

def mi_gg(cx, cy, biascorrect=True, demeaned=False):
    """Mutual information (MI) between two Gaussian variables in bits
   
    I = mi_gg(x,y) returns the MI between two (possibly multidimensional)
    Gassian variables, x and y, with bias correction.
    If x and/or y are multivariate columns must correspond to samples, rows
    to dimensions/variables. (Samples last axis) 
                                                                             
    biascorrect : true / false option (default true) which specifies whether
    bias correction should be applied to the esimtated MI.
    demeaned : false / true option (default false) which specifies whether th
    input data already has zero mean (true if it has been copula-normalized)

    """
    
    x = np.atleast_2d(cx)
    y = np.atleast_2d(cy)
    if x.ndim > 2 or y.ndim > 2:
        raise ValueError("x and y must be at most 2d")
    Ntrl = x.shape[1]
    Nvarx = x.shape[0]
    Nvary = y.shape[0]
    Nvarxy = Nvarx+Nvary

    if y.shape[1] != Ntrl:
        raise ValueError("number of trials do not match")

    # joint variable
    xy = np.vstack((x,y))
    # if not demeaned:
    #     xy = xy - xy.mean(axis=1)[:,np.newaxis]
    Cxy = np.dot(xy,xy.T) / float(Ntrl - 1)
    # submatrices of joint covariance
    Cx = Cxy[:Nvarx,:Nvarx]
    Cy = Cxy[Nvarx:,Nvarx:]

    chCxy = np.linalg.cholesky(Cxy)
    chCx = np.linalg.cholesky(Cx)
    chCy = np.linalg.cholesky(Cy)

    # entropies in nats
    # normalizations cancel for mutual information
    HX = np.sum(np.log(np.diagonal(chCx))) # + 0.5*Nvarx*(np.log(2*np.pi)+1.0)
    HY = np.sum(np.log(np.diagonal(chCy))) # + 0.5*Nvary*(np.log(2*np.pi)+1.0)
    HXY = np.sum(np.log(np.diagonal(chCxy))) # + 0.5*Nvarxy*(np.log(2*np.pi)+1.0)

    ln2 = np.log(2)
    if biascorrect:
        psiterms = sp.special.psi((Ntrl - np.arange(1,Nvarxy+1)).astype(float)/2.0) / 2.0
        dterm = (ln2 - np.log(Ntrl-1.0)) / 2.0
        HX = HX - Nvarx*dterm - psiterms[:Nvarx].sum()
        HY = HY - Nvary*dterm - psiterms[:Nvary].sum()
        HXY = HXY - Nvarxy*dterm - psiterms[:Nvarxy].sum()

    # MI in bits
    I = (HX + HY - HXY) / ln2
    return I
    
def gcmi_cc(epo1: mne.Epochs, epo2: mne.Epochs):
    """Gaussian-Copula Mutual Information between two continuous variables.

    I = gcmi_cc(x,y) returns the MI between two (possibly multidimensional)
    continuous variables, x and y, estimated via a Gaussian copula.
    If x and/or y are multivariate columns must correspond to samples, rows
    to dimensions/variables. (Samples first axis) 
    This provides a lower bound to the true MI value.

    """
    
    a = epo1.get_data(copy=False)[0][0][:]
    b = epo2.get_data(copy=False)[0][0][:]
    
    x = np.atleast_2d(a) # instead of shape (501,) --> (1, 501), x.ndim = 2
    y = np.atleast_2d(b)
        
    # Ntrl = x.shape[1] # number of trials
    # Nvarx = x.shape[0] # number of variables = 1
    # Nvary = y.shape[0]

    # if y.shape[1] != Ntrl:
    #     raise ValueError("number of trials do not match")

    # check for repeated values {loop through Nvarx but there is only 1 in our case!}
    # for xi in range(Nvarx):
    #     if (np.unique(x[xi,:]).size / float(Ntrl)) < 0.9:
    #         warnings.warn("Input x has more than 10% repeated values")
    #         break
    # for yi in range(Nvary):
    #     if (np.unique(y[yi,:]).size / float(Ntrl)) < 0.9:
    #         warnings.warn("Input y has more than 10% repeated values")
    #         break

    # copula normalization
    cx = copnorm(x)
    cy = copnorm(y)
    
    # parametric Gaussian MI
    I = mi_gg(cx,cy,True,True)
    
    return I

#CMI / TE?



# =============================================================================
# k-NEAREST NEIGHBOUR: MUTUAL INFORMATION, CONDITIONAL MUTUAL INFORMATION
# =============================================================================

def knn_mi():
    return None

def knn_cmi():
    # check https://github.com/omesner/knncmi/blob/master/knncmi/knncmi.py
    return None







# =============================================================================
# VISUALISATION
# =============================================================================


def plot_entropy_mi(epo1: mne.Epochs, epo2: mne.Epochs, entropy_mi: np.array):
    
    '''
    Parameters
    ----------
    epo1 : mne.EpochsFIF
        Signal 1.
    epo2 : mne.EpochsFIF
        Signal 2.
    entropies : np.array
        Array of 3 columns representing H(X), H(Y), H(X,Y) per channel..

    Returns
    -------
    None.

    '''
    
    avg_ent_p1, avg_ent_p2, avg_jent = entropy_mi[:, 0], entropy_mi[:, 1], entropy_mi[:, 2]
    
    montage = make_standard_montage('standard_1020')
    info1 = mne.create_info(epo1.info.ch_names, sfreq=1, ch_types='eeg').set_montage(montage)
    info2 = mne.create_info(epo2.info.ch_names, sfreq=1, ch_types='eeg').set_montage(montage)
    
    # Create an Evoked object with average entropy values
    evoked1 = mne.EvokedArray(avg_ent_p1[:, np.newaxis], info1)
    evoked2 = mne.EvokedArray(avg_ent_p2[:, np.newaxis], info2)
    
    
    fig, axes = plt.subplots(1,2,figsize=(15,6))
    plt.subplots_adjust(wspace=.001)
    
    img1,_ = mne.viz.plot_topomap(avg_ent_p1, evoked1.info, ch_type = 'eeg', cmap='viridis', 
                                  vlim=[np.min(entropy_mi[:,:2]), np.max(entropy_mi[:,:2])], 
                                  show=False, sensors=True, contours=0, axes=axes[0])
    axes[0].set_xlabel('Participant 1', fontsize=15)
    
    img2,_ = mne.viz.plot_topomap(avg_ent_p2, evoked2.info, ch_type = 'eeg', cmap='viridis', 
                                  vlim=[np.min(entropy_mi[:,:2]), np.max(entropy_mi[:,:2])], 
                                  show=False, sensors=True, contours=0, axes=axes[1])
    axes[1].set_xlabel('Participant 2', fontsize=15)
    
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes('right', size='10%', pad=.5)
    cbar = plt.colorbar(img2, cax=cax, orientation='vertical')
    cbar.set_label('Entropy')
    
    plt.suptitle('Topomaps of Average Shannon Entropy', fontsize=25)
    plt.show()
        
    
    
    fig, ax = plt.subplots()
    
    img3,_ = mne.viz.plot_topomap(avg_jent, evoked1.info, ch_type='eeg', cmap='plasma',
                                  vlim=[0, np.max(avg_jent)], show=False, sensors=True,
                                  contours=0, axes=ax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='10%', pad=.5)
    cbar = plt.colorbar(img3, cax=cax, orientation='vertical')
    cbar.set_label('Mutual Information')
    
    plt.suptitle('Topomaps of Average Mutual Information', fontsize=25)
    plt.show()






epo1 = mne.read_epochs(Path('data/participant1-epo.fif').resolve(), preload=True)
epo2 = mne.read_epochs(Path('data/participant2-epo.fif').resolve(), preload=True)
mne.epochs.equalize_epoch_counts([epo1, epo2])


# mi_hist_value = mi_hist(epo1, epo2, vis=True) #MI values per channel!
# mi_symb(epo1, epo2, vis=True)
# for l in range(1,5):
#     for m in range(1,5):
#         print('l =',l,', m =',m)
#         print('TE:', te_symb(epo1, epo2, l, m))
               
te_symb(epo1,epo2,3,3)

cmi_hist(epo1, epo2, 3, 3)           
