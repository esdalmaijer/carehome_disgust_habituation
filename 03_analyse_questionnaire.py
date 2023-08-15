#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

from matplotlib import pyplot
import numpy
import scipy.stats


# # # # #
# SETUP

# NORM DATA
tybur_2009 = { \
    "m": { \
        "pathogen": 3.87, \
        "sexual": 3.31, \
        "moral": 3.70, \
        }, \

    "sd": { \
        "pathogen": 1.19, \
        "sexual": 1.52, \
        "moral": 1.14, \
        }, \

    "n": { \
        "pathogen": 507, \
        "sexual": 507, \
        "moral": 507, \
        }, \
    }

# FILES AND FOLDERS
# Path the this file's directory.
DIR = os.path.dirname(os.path.abspath(__file__))
# Path to the data directory.
DATADIR = os.path.join(DIR, "data_extracted")
if not os.path.isdir(DATADIR):
    raise Exception("ERROR: Could not find data directory at '%s'" % (DATADIR))
OUTDIR = os.path.join(DIR, "output")
if not os.path.isdir(OUTDIR):
    os.mkdir(OUTDIR)


# # # # #
# LOAD DATA

# Load the reduced qualtrics data for the carehome group.
fpath = os.path.join(DATADIR, "carehome_qualtrics_data_reduced.csv")
raw = numpy.loadtxt(fpath, dtype=str, delimiter=",", unpack=True)

# Parse the data into a more readable format.
data = {}
data["carehome"] = {}
for i in range(raw.shape[0]):
    var = raw[i,0]
    val = raw[i,1:]
    try:
        data["carehome"][var] = val.astype(float)
    except:
        data["carehome"][var] = val

# Load the extracted TDDS data for the control group.
fpath = os.path.join(DATADIR, "control_tdds.csv")
raw = numpy.loadtxt(fpath, dtype=str, delimiter=",", unpack=True)

# Parse the data into a more readable format.
data["control"] = {}
for i in range(raw.shape[0]):
    var = raw[i,0]
    val = raw[i,1:]
    try:
        data["control"][var] = val.astype(float)
    except:
        data["control"][var] = val


# # # # #
# FREQUENCY PLOTS

# Plot and data settings.
situations = { \
    "adult_pads": {"category": "core", "colour":"#ce5c00"}, \
    "soiled_adult_pads": {"category": "core", "colour":"#a40000"}, \
    "faeces_outside_toilet": {"category": "core", "colour":"#2e3436"}, \
    "catheter_bags": {"category": "core", "colour":"#c4a000"}, \
    "stoma_bags": {"category": "core", "colour":"#8f5902"}, \
    "vomit": {"category": "core", "colour":"#4e9a06"}, \

    "oral_care": {"category": "gore", "colour":"#5c3566"}, \
    "blood_stains_clothing_bedding_floor_furniture": {"category": "gore", "colour":"#a40000"}, \
    "open_wounds": {"category": "gore", "colour":"#8f5902"}, \
    }
frequencies = ["Less than once per week", "Once or twice per week", \
    "Three to five times per week", "Five to ten times per week", \
    "Ten to twenty times per week", "Thirty to fifty times per week", \
    "More than fifty times per week"]
xticklabels = ["< 1", "1-2", "3-5", "5-10", "10-20", "30-50", "50+"]
xticks = range(1, 1+len(xticklabels))
# Determine the bar width.
n_situations = {}
for key in situations:
    if situations[key]["category"] not in n_situations.keys():
        n_situations[situations[key]["category"]] = 0
    n_situations[situations[key]["category"]] += 1
bar_width = 0.8 / max(n_situations.values())

# Plot the frequency of disgusting occurences.
fig, ax = pyplot.subplots(nrows=2, ncols=1, figsize=(8.27,8.0), dpi=900.0)
fig.subplots_adjust(top=0.98, bottom=0.08, left=0.1, right=0.99, hspace=0.2)
axes = {"core":ax[0], "gore":ax[1]}
count = {"core":0, "gore":0}

# Run through all situations.
for situation in situations.keys():

    # Get the data, and count the frequency of each occurence.
    var = "frequency_{}".format(situation)
    hist = numpy.zeros(len(frequencies), dtype=numpy.int64)
    for i, freq in enumerate(frequencies):
        hist[i] += numpy.sum(data["carehome"][var] == freq)
    hist = 100 * hist / numpy.sum(hist)
    
    # Get the disgust category and plot colour.
    cat = situations[situation]["category"]
    col = situations[situation]["colour"]
    
    # Create a more readable name.
    if situation == "blood_stains_clothing_bedding_floor_furniture":
        name = "blood stains"
    else:
        name = situation.replace("_", " ")
    name = name[0].upper() + name[1:]
    
    # Plot the occurences.
    #axes[cat].plot(xticks, hist, "-", lw=3, c=col, alpha=0.8, label=name)
    offset = -0.5 * n_situations[cat] * bar_width
    x = numpy.array(xticks) + count[cat] * bar_width + offset
    axes[cat].bar(x, hist, linewidth=2, width=bar_width, facecolor=col, \
        edgecolor="black", label=name, align="edge")

    # Increment the counter for this category.
    count[cat] += 1

# Finish the plots
for cat in axes.keys():
    axes[cat].set_xticks(xticks)
    axes[cat].set_xticklabels(xticklabels, fontsize=12)
    axes[cat].set_xlabel("Weekly occurrences of {} disgust".format(cat), \
        fontsize=16)
    axes[cat].set_ylim(-0.2, 55)
    yticklabels = axes[cat].get_yticks().astype(numpy.int64)
    axes[cat].set_yticklabels(yticklabels, fontsize=12)
    axes[cat].set_ylabel("Reported frequency (%)", fontsize=16)
    ncols = int(n_situations[cat] // 3 + numpy.ceil(n_situations[cat] % 3))
    axes[cat].legend(loc="upper right", ncol=ncols, fontsize=12)

# Save and close.
fig.savefig(os.path.join(OUTDIR, "fig-01_disgust_frequencies.png"))
pyplot.close(fig)



# # # # #
# TDDS REPORT

# Compute TDDS scores for subscales.
with open(os.path.join(OUTDIR, "TDDS_results.txt"), "w") as f:
    f.write("TDDS averages per group and subscale")
for sample in data.keys():
    for subscale in ["pathogen", "sexual", "moral"]:
        questions = []
        for qname in data[sample].keys():
            if qname[:4] == "tdds":
                if subscale in qname:
                    questions.append(qname)
        q = numpy.vstack([data[sample][qname] for qname in questions])
        data[sample]["tdds_{}".format(subscale)] = numpy.mean(q, axis=0)
        with open(os.path.join(OUTDIR, "TDDS_results.txt"), "a") as f:
            f.write("\n{}, {}: M={}, SD={}".format(sample, subscale, \
                round(numpy.mean(data[sample]["tdds_{}".format(subscale)]),2), \
                round(numpy.std(data[sample]["tdds_{}".format(subscale)]), 2)))
    if sample == "carehome":
        with open(os.path.join(OUTDIR, "TDDS_results.txt"), "a") as f:
            f.write("\n{}, {}: M={}, SD={}".format(sample, "months worked", \
                round(numpy.mean(data[sample]["carehome_career_duration_months"]),2), \
                round(numpy.std(data[sample]["carehome_career_duration_months"]), 2)))

# Comparisons of sub-groups against norm scores.
DIFFERENCE_TESTS = ( \
    ("carehome", "control"), \
    ("carehome", "norm"), \
    ("control", "norm"), \
    )
for sample1, sample2 in DIFFERENCE_TESTS:
    with open(os.path.join(OUTDIR, "TDDS_results.txt"), "a") as f:
        f.write("\n\nWelch's test of differences")
    for subscale in ["pathogen", "sexual", "moral"]:

        # Compute N, M, and SD for the first sample group.
        n1 = data[sample1]["tdds_{}".format(subscale)].shape[0]
        m1 = numpy.mean(data[sample1]["tdds_{}".format(subscale)])
        sd1 = numpy.sum((data[sample1]["tdds_{}".format(subscale)] - m1)**2) \
            / (n1-1)
        sem1 = sd1 / numpy.sqrt(n1)

        # Get N, M, and SD from the norm group.
        if sample2 == "norm":
            n2 = tybur_2009["n"][subscale]
            m2 = tybur_2009["m"][subscale]
            sd2 = tybur_2009["sd"][subscale]
            sem2 = sd2 / numpy.sqrt(n2)
        # Compute N, M, and SD for the second sample group.
        else:
            n2 = data[sample2]["tdds_{}".format(subscale)].shape[0]
            m2 = numpy.mean(data[sample2]["tdds_{}".format(subscale)])
            sd2 = numpy.sum((data[sample2]["tdds_{}".format(subscale)] - m2)**2) \
                / (n2-1)
            sem2 = sd2 / numpy.sqrt(n2)

        # Welch's test.
        t = (m1 - m2) / numpy.sqrt(sem1**2 + sem2**2)
        v = ((sd1**2 / n1) + (sd2**2 / n2))**2 \
            / ((sd1**4 / (n1**2*(n1-1))) + (sd2**4 / (n2**2*(n2-1))))
        p = 2 * (1 - scipy.stats.t.cdf(abs(t), v))

        # Write results to file.
        with open(os.path.join(OUTDIR, "TDDS_results.txt"), "a") as f:
            f.write("\n{}-{}, {}: t({})={}, p={}".format(sample1, sample2, \
                subscale, round(v,2), round(t,2), round(p,7)))

# Correlate subscales and duration of work.
with open(os.path.join(OUTDIR, "TDDS_results.txt"), "a") as f:
    f.write("\n\nKendall's correlation coefficient for the carehome group")
for subscale in ["pathogen", "sexual", "moral"]:
    x = data["carehome"]["carehome_career_duration_months"]
    y = data["carehome"]["tdds_{}".format(subscale)]
    tau, p = scipy.stats.kendalltau(x, y)
    with open(os.path.join(OUTDIR, "TDDS_results.txt"), "a") as f:
        f.write("\n{} x {}: tau={}, p={}".format(subscale, \
            "carehome_career_duration_months", round(tau,2), round(p,3)))
