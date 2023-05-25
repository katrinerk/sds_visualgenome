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
import configparser


import vgiterator
from sds_input_util import VGSentences, VGParam
from vgindex import VgitemIndex
from vgpaths import VGPaths
from polysemy_util import SyntheticPolysemes


########3




parser = ArgumentParser()
parser.add_argument('--output', help="directory to write output to, default: sds_in/discourse", default = "sds_in/discourse/")
parser.add_argument('--vgdata', help="directory with VG data including frequent items, train/test split, topic model", default = "data/")
parser.add_argument('--poly', help="add synthetic polysemy? default: False", action = "store_true")

args = parser.parse_args()


# settings file
config = configparser.ConfigParser()
config.read("settings.txt")
selpref_method = config["Selpref"]


print("Selectional preference method:", selpref_method["Method"])

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


####
# write parameters for SDS.
print("computing parameters")
vgparam_obj = VGParam(vgpath_obj, selpref_method, frequentobj = vgobjects_attr_rel)


global_param, scenario_concept_param, word_concept_param, selpref_param = vgparam_obj.get()

#####
# compute sentences for SDS:
print("computing sentences")

vgsent_obj = VGSentences(vgpath_obj)

# # sentences, as they come out of vgsent_obj.each_sentence:
# # tuples (sentence ID, word literals, word literals that need to be kept, role literals)
paragraph= [ ]
counter = 0

for sentid, words, roles, in vgsent_obj.each_sentence(vgobj, vgobjects_attr_rel, traintest_split, "train"):
    paragraph.append([sentid, words, roles])
    counter += 1
    if counter > 3:
        break


##
# adding polysemy?
if args.poly:
    poly_obj = SyntheticPolysemes(vgpath_obj, vgindex_obj, vgobjects_attr_rel, scenario_concept_param)
    
    next_wordid = len(vgobjects_attr_rel["objects"]) + len(vgobjects_attr_rel["attributes"]) + len(vgobjects_attr_rel["relations"])
    paragraph_transformed, goldwords = poly_obj.make(paragraph, next_wordid)

    # adapting parameters for new words
    num_clozewords = len(goldwords)
    global_param["num_words"] += num_clozewords

    # # add to word-concept log probabilities:
    # # cloze word IDs -> concept ID -> logprob
    for word_id, entry in goldwords.items():

        # word cloze ID -> concept ID -> log of 0.5:
        # equal output probability for both concepts
        word_concept_param[ word_id ] = dict( (entry["concept_ids"][i],  -0.69) for i in [0,1])
    
else:
    paragraph_transformed = [ [sentid, words, [], roles] for sentid, words, roles in paragraph]

print("writing parameters")

vgparam_obj.write(global_param, scenario_concept_param, word_concept_param, selpref_param)

print("writing sentences")
vgsent_obj.write_discourse([paragraph_transformed])

