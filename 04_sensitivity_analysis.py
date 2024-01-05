#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import matplotlib
from matplotlib import pyplot
import numpy
import scipy.stats
from scipy.stats import kendalltau, pearsonr, spearmanr

def compute_h(n1, n2):
    return ((n1+n2-2) / n1) + ((n1+n2-2) / n2)
    
def r2d(r, n1, n2):
    h = compute_h(n1, n2)
    d = (numpy.sqrt(h) * r) / numpy.sqrt(1 - r**2)
    return d

def d2r(d, n1, n2):
    h = compute_h(n1, n2)
    r = d / numpy.sqrt(d**2 + h)
    return r


# # # # #
# SETTINGS

# Study settings.
N_CAREHOME = 32
N_CONTROL = 50

# Simulation settings.
N_RANGE = numpy.array([0, 10, 20, 30, 40, 50, 100, 200])
COV_RANGE = numpy.linspace(0.0, 1.0, 11)
D_RANGE = numpy.linspace(0.0, 1.0, 21)
R_RANGE = d2r(D_RANGE, 2, 2)
# R_RANGE = numpy.linspace(0.0, 1.0, 21) #d2r(D_RANGE, 2, 2)
# R_RANGE[-1] = 1.0 - 1e-6
# D_RANGE = r2d(R_RANGE, 33, 50)
N_REPETITIONS = 100
N_REPETITIONS_WELCH = 1000
N_REPETITIONS_CORRELATION = 1000
ALPHA = 0.05

# Directory settings.
DIR = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(DIR, "output")
if not os.path.isdir(OUTDIR):
    os.mkdir(OUTDIR)


# # # # #
# SENSITIVITY

# Pre-create variables to hold results in.
result = {}
for var in ["t", "df", "p", "r-n1", "r_p-n1", "r-n2", "r_p-n2"]:
    result[var] = numpy.zeros((len(D_RANGE), N_REPETITIONS_WELCH), \
        dtype=numpy.float64) * numpy.NaN
for var in ["r", "r_p", "rho", "rho_p", "tau", "tau_p"]:
    result[var] = numpy.zeros((2, len(R_RANGE), N_REPETITIONS_WELCH), \
        dtype=numpy.float64) * numpy.NaN
power = {}
for var in ["t"]:
    power[var] = numpy.zeros(len(D_RANGE), dtype=numpy.float64) * numpy.NaN
for var in ["r", "rho", "tau"]:
    power[var] = numpy.zeros((2,len(R_RANGE)), dtype=numpy.float64) * numpy.NaN

# Copy the group sizes.
n1 = N_CAREHOME
n2 = N_CONTROL
# Run through all effect sizes we need to test.
for di, d in enumerate(D_RANGE):
    # Set the group means based on the effect size.
    m1_truth = 0.0 - d/2
    m2_truth = 0.0 + d/2
    # Run through the required number of repetitions.
    for i in range(N_REPETITIONS_WELCH):
        # Create psuedo-random data.
        X = numpy.random.randn(n1+n2)
        y = numpy.zeros(n1+n2, dtype=numpy.int64)
        y[n1:] = 1
        X[y==0] += m1_truth
        X[y==1] += m2_truth
        # Compute means.
        m1 = numpy.nanmean(X[y==0])
        m2 = numpy.nanmean(X[y==1])
        # Compute SDs (should be near 1, as we're using standardised space).
        sd1 = numpy.nanstd(X[y==0])
        sd2 = numpy.nanstd(X[y==1])
        # Compute standard errors.
        sem1 = sd1 / numpy.sqrt(n1)
        sem2 = sd2 / numpy.sqrt(n2)
        # Welch's test.
        t = (m1 - m2) / numpy.sqrt(sem1**2 + sem2**2)
        v = ((sd1**2 / n1) + (sd2**2 / n2))**2 \
            / ((sd1**4 / (n1**2*(n1-1))) + (sd2**4 / (n2**2*(n2-1))))
        p = 2 * (1 - scipy.stats.t.cdf(abs(t), v))
        # Copy outcomes.
        result["t"][di,i] = t
        result["df"][di,i] = v
        result["p"][di,i] = p
    # Compute power.
    power["t"][di] = numpy.nansum(result["p"][di,:] < ALPHA) \
        / float(result["p"].shape[1])

# Set the group means.
m = numpy.zeros(2, dtype=numpy.float64)
# Run through both group sizes.
for ni, n in enumerate([N_CAREHOME, N_CONTROL]):
    # Run through all effect sizes we need to test.
    for ci, cov in enumerate(R_RANGE):
        for i in range(N_REPETITIONS_CORRELATION):
            # Simulate data.
            cov_matrix = [[1.0, cov], [cov, 1.0]]
            X = numpy.random.multivariate_normal(m, cov_matrix, size=n)
            x = X[:,0]
            y = X[:,1]
            # Compute correlation coefficients.
            result["r"][ni, ci, i], result["r_p"][ni, ci, i] = \
                pearsonr(x, y)
            result["rho"][ni, ci, i], result["rho_p"][ni, ci, i] = \
                spearmanr(x, y)
            result["tau"][ni, ci, i], result["tau_p"][ni, ci, i] = \
                kendalltau(x, y)
        # Compute power for this covariance.
        power["r"][ni,ci] = numpy.sum(result["r_p"][ni, ci, :] < ALPHA) \
            / float(result["r"].shape[2])
        power["rho"][ni,ci] = numpy.sum(result["rho_p"][ni, ci, :] < ALPHA) \
            / float(result["rho"].shape[2])
        power["tau"][ni,ci] = numpy.sum(result["tau_p"][ni, ci, :] < ALPHA) \
            / float(result["tau"].shape[2])

# Record power per effect size.
with open(os.path.join(OUTDIR, "power_for_Welch.csv"), "w") as f:
    f.write(",".join(["d", "power"]))
    for di, d in enumerate(D_RANGE):
        line = [d, power["t"][di]]
        f.write("\n" + ",".join(map(str, line)))

with open(os.path.join(OUTDIR, "power_for_correlation.csv"), "w") as f:
    header = ["cov"]
    for ni, n in enumerate([N_CAREHOME, N_CONTROL]):
        for var in ["r", "rho", "tau"]:
            header.append("{}_n={}".format(var, n))
    f.write(",".join(header))

    for ci, cov in enumerate(R_RANGE):
        line = [cov]
        for ni, n in enumerate([N_CAREHOME, N_CONTROL]):
            for var in ["r", "rho", "tau"]:
                line.append(power[var][ni, ci])
        f.write("\n" + ",".join(map(str, line)))

# Create a new figure.
fig, axes = pyplot.subplots(nrows=1, ncols=1, figsize=(8.0, 6.0), dpi=900.0)
# Adjust the buffer areas and distances between subplots.
fig.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95, \
    wspace=0.2, hspace=0.2)
# Plot the power for each coefficient separately.
for oi, outcome in enumerate(["t", "r_0", "r_1"]):
    # Choose the variables for this outcome.
    if outcome == "t":
        val = power["t"]
        es_range = D_RANGE
        # es_range = d2r(D_RANGE, N_CAREHOME, N_CONTROL)
        col = "#4e9a06"
        lbl = "Welch's test (N={}, N={})".format(N_CAREHOME, N_CONTROL)
    elif outcome[:2] == "r_":
        outcome, ni = outcome.split("_")
        ni = int(ni)
        n = [N_CAREHOME, N_CONTROL][ni]
        val = power["r"][ni,:]
        es_range = r2d(R_RANGE, n, n)
        # es_range = R_RANGE
        col = ["#ce5c00", "#c4a000"][ni]
        lbl = "Correlation (N={})".format(n)
    # Convenience renaming of the axis.
    ax = axes
    # Plot the power for each effect size.
    ax.plot(es_range, val, "o-", lw=3, color=col, label=lbl)
    # Annotate the R values.
    if (outcome == "r") and (n == N_CAREHOME):
        for i, r in enumerate(R_RANGE):
            if (es_range[i] > 0.1) and (es_range[i] < 0.95):
                r_txt = "R={}".format(round(r, 2)).ljust(6, "0")
                ax.annotate(r_txt, (es_range[i]+0.01, val[i]-0.02), \
                    color=col, fontsize=10, rotation=0)
    # Finish the plot.
    ax.set_ylim([0, 1])
    ax.set_ylabel("Power", fontsize=16)
    ax.set_xlim([0, 1])
    ax.set_xlabel("Effect size (d)", fontsize=16)
    ax.legend(loc="lower right", fontsize=12)
# Save and close the figure.
fig.savefig(os.path.join(OUTDIR, "fig-s01_sensitivity_plot.png"))
pyplot.close(fig)

