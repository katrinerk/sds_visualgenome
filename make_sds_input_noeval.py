# Katrin Erk January 2023
# Write input for an SDS system:
# parameter files, and input sentences
#
# Here: input without any evaluation, no ambiguity at word level,
# only at scenario level


import sys
import os
import json
import zipfile
from collections import defaultdict, Counter
import nltk
import math
import numpy as np
import random

import text_histogram 
import vgiterator
from sds_input_util import VGSentences, VGParam
from vgindex import VgitemIndex
from vgpaths import VGPaths


########3

vgpath_obj = VGPaths()

print("reading data")

vgcounts_zipfilename, vgcounts_filename = vgpath_obj.vg_counts_zip_and_filename()
with zipfile.ZipFile(vgcounts_zipfilename) as azip:
    with azip.open(vgcounts_filename) as f:
        vgobjects_attr_rel = json.load(f)
vgindex_obj = VgitemIndex(vgobjects_attr_rel)

# obtain IDs of training and test images

split_zipfilename, split_filename = vgpath_obj.vg_traintest_zip_and_filename()
with zipfile.ZipFile(split_zipfilename) as azip:
    with azip.open(split_filename) as f:
        traintest_split = json.load(f)
    
trainset = set(traintest_split["train"])

vgobj = vgiterator.VGIterator()

####
# write parameters for SDS.
# the only thing we need to change are the concept/word probabilities
vgparam_obj = VGParam(vgpath_obj, frequentobj = vgobjects_attr_rel)

print("computing parameters")

global_param, scenario_concept_param, word_concept_param, selpref_param = vgparam_obj.get()

print("writing parameters")
vgparam_obj.write(global_param, scenario_concept_param, word_concept_param, selpref_param)

#####
# write sentences for SDS:
print("computing sentences")

vgsent_obj = VGSentences(vgpath_obj)

# sentences: tuples (sentence ID, word literals, word literals that need to be kept, role literals)
sentences = [ ]
for sentid, words, roles in vgsent_obj.each_testsentence(vgobj, vgobjects_attr_rel, traintest_split):
    sentences.append( [sentid, words, [], roles])

print("writing sentences")
vgsent_obj.write(sentences)

