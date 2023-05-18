# Katrin Erk May 2023
# Write input for an SDS system:
# parameter files, and input sentences
#
# Here: multi-sentence discourse, with mental files.
# identify referent for a given NP


import sys
import os
import json
import zipfile
from collections import defaultdict, Counter
import math
import numpy as np
from argparse import ArgumentParser
import random

import vgiterator
from sds_input_util import VGSentences, VGParam
from vgindex import VgitemIndex
from vgpaths import VGPaths


########3




parser = ArgumentParser()
parser.add_argument('--output', help="directory to write output to, default: sds_in/discourse", default = "sds_in/discourse/")
parser.add_argument('--vgdata', help="directory with VG data including frequent items, train/test split, topic model", default = "data/")
parser.add_argument('--selpref_relfreq', help="selectional preferences using relative frequency rather than similarity to centroid?  default: False", action = "store_true")

args = parser.parse_args()

print("Using embeddings to compute selectional constraints?", not(args.selpref_relfreq))

vgpath_obj = VGPaths(vgdata = args.vgdata,  sdsdata = args.output)

print("reading data")

# frequent obj/attr/rel
vgcounts_zipfilename, vgcounts_filename = vgpath_obj.vg_counts_zip_and_filename()
with zipfile.ZipFile(vgcounts_zipfilename) as azip:
    with azip.open(vgcounts_filename) as f:
        vgobjects_attr_rel = json.load(f)
        
vgindex_obj = VgitemIndex(vgobjects_attr_rel)

# training/test split
split_zipfilename, split_filename = vgpath_obj.vg_traintest_zip_and_filename()
with zipfile.ZipFile(split_zipfilename) as azip:
    with azip.open(split_filename) as f:
        traintest_split = json.load(f)
    
trainset = set(traintest_split["train"])

vgobj = vgiterator.VGIterator(vgcounts = vgobjects_attr_rel)

####
# write parameters for SDS.
vgparam_obj = VGParam(vgpath_obj, frequentobj = vgobjects_attr_rel,
                      selpref_vectors = not(args.selpref_relfreq))

print("computing parameters")

global_param, scenario_concept_param, word_concept_param, selpref_param = vgparam_obj.get()
vgparam_obj.write(global_param, scenario_concept_param, word_concept_param, selpref_param)

#####
# write sentences for SDS:
print("computing sentences")

vgsent_obj = VGSentences(vgpath_obj)

# sentences, as they come out of vgsent_obj.each_sentence:
# tuples (sentence ID, word literals, word literals that need to be kept, role literals)
paragraph= [ ]
counter = 0

for sentid, words, roles, in vgsent_obj.each_sentence(vgobj, vgobjects_attr_rel, traintest_split, "train"):
    paragraph.append([sentid, words, [], roles])
    counter += 1
    if counter > 3:
        break


print("writing sentences")
vgsent_obj.write_discourse([paragraph])

