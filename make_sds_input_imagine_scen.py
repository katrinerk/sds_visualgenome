# Katrin Erk February 2023
# Write input for an SDS system:
# parameter files, and input sentences
#
# Here: input for the scenario enrichment/imagination test


import sys
import os
import json
import zipfile
from collections import defaultdict, Counter
import nltk
import math
import numpy as np
import random

import vgiterator
from sds_input_util import VGSentences, VGParam
from vgindex import VgitemIndex
from vgpaths import VGPaths

num_testsentences = 20
max_num_objects_tokeep = 25
shortsentence_fraction_test = 0.3

##################
# read data

vgpath_obj = VGPaths(sdsdata = "sds_in/imagine_scen")

# read frequent objects, attributes, relations
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

random.seed(6543)


#########

##########################
# write parameters for SDS.
# nothing to be changed

print("writing SDS parameters")
    
vgparam_obj = VGParam(vgpath_obj)
global_param, scenario_concept_param, word_concept_param, selpref_param = vgparam_obj.get()
vgparam_obj.write(global_param, scenario_concept_param, word_concept_param, selpref_param)

########################
# write sentences for SDS:
# retain only given number of test sentences, chosen at random
#
# transform sentences:
# for each test sentence, hide
# shortsentence_fraction_test of the objects (for short sentences)
# or all objects beyond max_num_objects_tokeep

print("Writing sentences")

vgsent_obj = VGSentences(vgpath_obj)

# store all sentences so we can randomly select among them
sentences = [ ]

for sentid, words, roles in vgsent_obj.each_testsentence(vgobj, vgobjects_attr_rel, traintest_split):
    sentences.append( (sentid, words, roles) )


# randomly select test sentences
testsentences = random.sample(sentences, num_testsentences)

# hide some objects in each test sentence,
# store in gold object what we hid
testsentences_transformed = [ ]

gold = { "imagine_scen" : { }}

for sentid, words, roles in testsentences:
    # determine which unary literals refer to objects,
    # as we are only downsampling those
    object_literals = [ (w, labelid, dref) for w, labelid, dref in words if vgindex_obj.ix2l(labelid)[1] == "obj"]

    num_to_remove = len(object_literals) - max_num_objects_tokeep if len(object_literals) > max_num_objects_tokeep else int(len(object_literals) * shortsentence_fraction_test)

    if num_to_remove == 0:
        # nothing left to remove, don't use this sentence
        continue
    
    literals_to_remove = random.sample(object_literals, num_to_remove)
    words, roles = vgsent_obj.remove_from_sent(words, roles, literals = literals_to_remove)

    # store the remaining sentence
    testsentences_transformed.append( (sentid, words, [ ], roles) )

    # and record, but only object labels
    gold["imagine_scen"][sentid] = [ labelid for _, labelid, _ in literals_to_remove]


        
###
# write sentences to file
vgsent_obj.write(testsentences_transformed)


###
# write gold information
gold_zipfile, gold_file = vgpath_obj.sds_gold()
with zipfile.ZipFile(gold_zipfile, "w", zipfile.ZIP_DEFLATED) as azip:
    azip.writestr(gold_file, json.dumps(gold))

