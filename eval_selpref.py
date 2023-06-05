# Katrin Erk May 2023
# in-vitro evaluation of selectional preferences approaches
# used in SDS


import sys
import os
import json
import zipfile
from collections import defaultdict, Counter
import math
import numpy as np
import random
from argparse import ArgumentParser
import configparser

from scipy import stats

from vgnames import VGOBJECTS, VGATTRIBUTES, VGRELATIONS 
import vgiterator
import sentence_util
from vgpaths import VGPaths, get_output_path
from vgindex import VgitemIndex
from polysemy_util import SyntheticPolysemes
from sds_input_util import VGSentences, VGParam


###
# input

parser = ArgumentParser()
parser.add_argument('--vgdata', help="directory with VG data including frequent items, train/test split, topic model", default = "data/")
parser.add_argument('--test', help = "evaluate on test sentences rather than dev. Default: false.", action = "store_true")
parser.add_argument('--numpts', help="number of predicate/role pairs to sample for evaluation, default 3000", type = int, default = 3000)
parser.add_argument('--simlevel', help="choose cloze pairs only in this similarity range: 0-3, 0 is most similar, default: no restriction", type =int, default = -1)


args = parser.parse_args()

# settings file
config = configparser.ConfigParser()
config.read("settings.txt")
selpref_method = config["Selpref"]

#################3
print("reading data")

random.seed(48239)

vgpath_obj = VGPaths(vgdata = args.vgdata)

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
# compute parameters for SDS (we really only need the selectional preferences)
print("computing parameters")
vgparam_obj = VGParam(vgpath_obj, selpref_method, frequentobj = vgobjects_attr_rel)


global_param, scenario_concept_param, word_concept_param, selpref_param = vgparam_obj.get()


####
# read test sentences
vgiter = vgiterator.VGIterator()
vgsent_obj = VGSentences(vgpath_obj)

sentences = [ ]

if args.test:
    section = "test"
else:
    section = "dev"

for sentid, words, roles in vgsent_obj.each_sentence(vgiter, vgobjects_attr_rel, traintest_split, section):
    if len(roles) > 0:
        sentences.append( (sentid, words, roles) )

############3
# in selectional preferences, look up predicate/argument pairs
# shape of selpref_param:
# { "arg0" : {"config" : list of pairs (pred index, arg index), "weight" : list of weights for the pairs},
#   "arg1" : {"config" : list of pairs as above, "weight" : list of weights as above } }
#
# reformat paramters to have the shape
# rolelabel -> predicate index -> argument index -> weight
selpref_lookup = {}
for rolelabel in ["arg0", "arg1"]:
    
    selpref_lookup[rolelabel] = defaultdict(dict)
    
    for predarg, weight in zip(selpref_param[rolelabel]["config"], selpref_param[rolelabel]["weight"]):
        predid, argid = predarg
        selpref_lookup[rolelabel][predid][argid] = weight

##########
# numpts times: sample a test sentence, sample a relation, sample a cloze partner, predict, evaluate accuracy
print("evaluating")

# store correctness of each prediction as 1 or 0
correct = [ ]

# how many datapoints have we processed?
counter = 0

# make cloze of argument
poly_obj = SyntheticPolysemes(vgpath_obj, vgindex_obj, vgobjects_attr_rel, scenario_concept_param)
next_wordid = len(vgobjects_attr_rel[VGOBJECTS]) + len(vgobjects_attr_rel[VGATTRIBUTES]) + len(vgobjects_attr_rel[VGRELATIONS])

while True:
    # sample a test sentence
    sentence_index = random.choice(list(range(len(sentences))))
    sentid, words, roles = sentences[ sentence_index]
    
    # sample a binary literal, and take it apart
    roleliteral = random.choice(roles)
    _, rolelabel, drefh, drefd = roleliteral

    predliterals = [w for w in words if w[2] == drefh]
    argliterals = [w for w in words if w[2] == drefd]

    if len(predliterals) != 1:
        continue
    predlit= predliterals[0]
    predid = predlit[1]
    predlabel = vgindex_obj.ix2l(predid)[0]
    if predid not in selpref_lookup[rolelabel]:
        # no selectional preferences for this predicate and role
        continue

    if len(argliterals) < 0:
        continue
    _, argid, _ = argliterals[0]

    # word1id is argconceptid, word1 is the label of word1id,
    # word2id is the confounder, and word2 is its label
    retv = poly_obj.make_obj_cloze(argid, args.simlevel)
    if retv is None:
        continue

    # at this point we do have data to evaluate
    arglabel, argid, confounderlabel, confounderid = retv
    counter += 1

    # looking up arg and confounder in the selectional preferences
    argweight = selpref_lookup[rolelabel][predid].get(argid, None)
    confounderweight = selpref_lookup[rolelabel][predid].get(confounderid, None)

    # if both weights are None, that is, we couldn't look up either, then count this as an error

    if argweight is not None and (confounderweight is None or argweight > confounderweight):
        correct.append(1)
    else:
        correct.append(0)

    # print("predicate",predlabel, "argument", arglabel, argweight, "confounder", confounderlabel, confounderweight, "correctness", correct[-1])

    # end cycle if we've found sufficiently many
    if counter >= args.numpts:
        break


print()
print("Accuracy:", round(sum(correct) / len(correct), 3), 'That is',  sum(correct), "out of", len(correct))
