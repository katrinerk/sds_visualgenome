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
import math
import numpy as np
from argparse import ArgumentParser
import random
import configparser


import vgiterator
from sds_input_util import VGSentences, VGParam
from vgindex import VgitemIndex
from vgpaths import VGPaths


########3




parser = ArgumentParser()
parser.add_argument('--output', help="directory to write output to, default: sds_in/vanilla", default = "sds_in/vanilla/")
parser.add_argument('--vgdata', help="directory with VG data including frequent items, train/test split, topic model", default = "data/")
parser.add_argument('--numsent', help="number of test sentences to sample, default 100", type = int, default = 100)

args = parser.parse_args()


# settings file
config = configparser.ConfigParser()
config.read("settings.txt")
selpref_method = config["Selpref"]

print("Selectional constraints method", selpref_method["Method"])

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
vgparam_obj = VGParam(vgpath_obj, selpref_method, frequentobj = vgobjects_attr_rel)

print("computing parameters")

global_param, scenario_concept_param, word_concept_param, selpref_param = vgparam_obj.get()

# print(text_histogram.histogram(selpref_param["arg0"]["weight"]))
# print("---------")
# print(text_histogram.histogram(selpref_param["arg1"]["weight"]))


print("writing parameters")
vgparam_obj.write(global_param, scenario_concept_param, word_concept_param, selpref_param)

#####
# write sentences for SDS:
print("computing sentences")

vgsent_obj = VGSentences(vgpath_obj)

# sentences: tuples (sentence ID, word literals, word literals that need to be kept, role literals)
sentences = [ ]
for sentid, words, roles in vgsent_obj.each_sentence(vgobj, vgobjects_attr_rel, traintest_split, "test"):
    sentences.append( [sentid, words, [], roles])

random.seed(100)
use_sentences = random.sample(sentences, args.numsent)

print("writing sentences")
vgsent_obj.write(use_sentences)

