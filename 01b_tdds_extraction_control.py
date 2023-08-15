#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import copy

import numpy
import matplotlib
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

from MouseViewParser.readers import gorilla


# # # # #
# CONSTANTS

# BASIC CONTROLS
# Names for data folders and files.
DATAFOLDER = "data_exp_96401-v4"
FILENAME = "data_exp_96401-v4_questionnaire-qjy2.csv"

# QUESTIONNAIRE
# Number of TDDS questions.
N_QUESTIONS = 21
# Construct the questions we need to extract data for.
QUESTIONS = ["tdds{}".format(i+1) for i in range(N_QUESTIONS)]
# Due to a WHOOPSIE, "tdds12" is actually "response-1".
qi = QUESTIONS.index("tdds12")
QUESTIONS[qi] = "response-1"
# SUBSCALES
TDDS = { \
    "pathogen": [3, 6, 9, 12, 15, 18, 21], \
    "sexual":   [2, 5, 8, 11, 14, 17, 20], \
    "moral":    [1, 4, 7, 10, 13, 16, 19], \
    }

# EXCLUSIONS
# Excluded participants.
EXCLUDED = [ \
    ]

# FILES AND FOLDERS.
DIR = os.path.dirname(os.path.abspath(__file__))
DATADIR = os.path.join(DIR, DATAFOLDER)
FILEPATH = os.path.join(DATADIR, FILENAME)
OUTDIR = os.path.join(DIR, "output")
if not os.path.isdir(OUTDIR):
    os.mkdir(OUTDIR)


# # # # #
# LOAD DATA.

# Start with an empty dictionary.
data = {}

# Loop through all lines of data.
with open(os.path.join(DATADIR, FILENAME), "r") as f:

    # Start without any info on the header and the indices of relevant columns.
    header = None
    pi = None
    qi = None
    ri = None

    # Loop through all lines in this file.
    for i, line in enumerate(f):
        
        # Remove the trailing newline.
        line = line.replace("\n", "")

        # Stop if we reached the end of the file.
        if line == "END OF FILE":
            break

        # Split by commas.
        line = line.split(",")

        # The first line is the header.
        if header is None:
            # Copy the header.
            header = line[:]
            # Extract the indices for data we need.
            pi = header.index("Participant Public ID")
            qi = header.index("Question Key")
            ri = header.index("Response")
            # Skip processing, as this is not a data line.
            continue
        
        # Only process lines with data for the questions we need.
        if line[qi] not in QUESTIONS:
            continue
        
        # Check if this is a new participant.
        if line[pi] not in data.keys():
            data[line[pi]] = numpy.zeros(N_QUESTIONS, dtype=numpy.float64) \
                * numpy.NaN
        
        # Store the response.
        i = QUESTIONS.index(line[qi])
        data[line[pi]][i] = int(line[ri])


# # # # #
# WRITE TO FILE

with open(os.path.join(OUTDIR, "control_tdds.csv"), "w") as f:
    
    # Construct a header.
    header = ["ppname"]
    for qname in QUESTIONS:
        # Revert the WHOOPSIE.
        if qname == "response-1":
            qname = "tdds12"
        # Get the number from the question name.
        qnr = int(qname.replace("tdds", ""))
        # Find which subscale this question was in.
        for subscale in TDDS.keys():
            if qnr in TDDS[subscale]:
                break
        # Construct the full question name with number and subscale.
        header.append("tdds_{}_{}".format(qnr, subscale))

    # Write the header to the file.
    f.write(",".join(header))
    
    # Loop through all data.
    for ppname in data.keys():
        line = [ppname] + list(data[ppname].astype(numpy.int64))
        f.write("\n" + ",".join(map(str,line)))
    
