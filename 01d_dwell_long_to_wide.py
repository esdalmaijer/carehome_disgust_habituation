#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import numpy
import pandas
import scipy.stats
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.stats.power import tt_solve_power
from statsmodels.tools.eval_measures import aic, bic
from sklearn.impute import KNNImputer

FNAME = "carehome_dwell_means_long"

# FILES AND FOLDERS
# Path the this file's directory.
DIR = os.path.dirname(os.path.abspath(__file__))
# Path to the data directory.
DATADIR = os.path.join(DIR, "data_extracted")
if not os.path.isdir(DATADIR):
    raise Exception("ERROR: Could not find data directory at '%s'" % (DATADIR))


# # # # #
# LOAD AND CONVERT

# Load long data.
file_path = os.path.join(DATADIR, FNAME+".csv")
X = pandas.read_csv(file_path)

# Specify important column names.
value = "dwell"
within_columns = ["environment", "disgust_type", \
    "repetition", "stimulus", "stim_nr"]

# Copy the data to avoid accidental overwrites.
X_ = X.copy()
# Pivot X into wide format.
Xw_df = X_.pivot_table(values=value, index="ppname", \
    columns=within_columns, fill_value = numpy.NaN, dropna=False)
# Grab the header, which is in arrays of unique combinations of 
# possible values in the comments
X_wide_header = list(Xw_df.columns.values)
# Grab the actual values.
X_wide = Xw_df.values

# Write to long format.
with open(os.path.join(DATADIR, "{}-to-wide.csv".format(FNAME)), "w") as f:
    header = ["ppname"]
    for unique_combination in X_wide_header:
        environment, disgust_type, repetition, stimulus_type, stim_nr = \
            unique_combination
        header.append("env-{}_type-{}_rep-{}_stim-{}_stimnr-{}".format( \
            environment, disgust_type, repetition, stimulus_type, stim_nr))
    f.write(",".join(map(str, header)))
    for i in range(X_wide.shape[0]):
        line = [Xw_df.index[i]]
        for j in range(X_wide.shape[1]):
            line.append(X_wide[i,j])
        f.write("\n" + ",".join(map(str, line)))

# Store in dwell variable.
environments = ["carehome", "outside"]
disgust_types = ["core", "gore"]
repetitions = list(range(4))
stim_types = ["affective", "neutral"]
stim_nrs = list(range(4))
n_participants = X_wide.shape[0]
shape = (n_participants, len(environments), len(disgust_types), \
    len(repetitions), len(stim_types), len(stim_nrs))
dwell = numpy.zeros(shape, dtype=numpy.float64)
for i in range(X_wide.shape[0]):
    for ei, environment in enumerate(environments):
        for dti, disgust_type in enumerate(disgust_types):
            for ri, repetition in enumerate(repetitions):
                for sti, stim_type in enumerate(stim_types):
                    for sni, stim_nr in enumerate(stim_nrs):
                        label = (environment, disgust_type, repetition, \
                            stim_type, stim_nr)
                        j = X_wide_header.index(label)
                        dwell[i,ei,dti,ri,sti,sni] = X_wide[i,j]

# Compute dwell difference between affective and neutral stimuli.
d = dwell[:,:,:,:,0,:] - dwell[:,:,:,:,1,:]
# Compute the average difference over all stimuli.
m = numpy.nanmean(d, axis=-1)
# Compute the average difference over all repetitions.
m = numpy.nanmean(m, axis=-1)
# Compute the average difference over all disgust types.
m = numpy.nanmean(m, axis=-1)
# Compute the average difference over all environments.
m = numpy.nanmean(m, axis=-1)

# Write to file.
fname = FNAME.replace("long", "disgust_avoidance") + ".csv"
with open(os.path.join(DATADIR, fname), "w") as f:
    header = ["ppname", "dwell_difference"]
    f.write(",".join(header))
    for i in range(m.shape[0]):
        line = [Xw_df.index[i], m[i]]
        f.write("\n" + ",".join(map(str, line)))
    