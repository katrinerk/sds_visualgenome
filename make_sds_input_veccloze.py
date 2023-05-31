# Katrin Erk February 2023
# Write input for an SDS system:
# parameter files, and input sentences
#
# Here: input for the Cloze test
# between similar relations/attributes


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

from vgnames import VGOBJECTS, VGATTRIBUTES, VGRELATIONS 
import vgiterator
from sds_input_util import VGSentences, VGParam
from vgindex import VgitemIndex
from vec_util import VectorInterface
from polysemy_util import SyntheticPolysemes

from vgpaths import VGPaths


########3

parser = ArgumentParser()
parser.add_argument('--output', help="directory to write output to, default: sds_in/veccloze", default = "sds_in/veccloze/")
parser.add_argument('--vgdata', help="directory with VG data including frequent items, train/test split, topic model", default = "data/")
parser.add_argument('--numsent', help="Number of sentences to use, default 2000", type = int, default = 2000)
parser.add_argument('--maxlen', help="maximum sentence length, default = 25", type = int, default = 25)
parser.add_argument('--simlevel', help="choose cloze pairs only in this similarity range: 0-3, 0 is most similar, default: no restriction", type =int, default = -1)
parser.add_argument('--singleword', help="only one ambiguous word per sentence. default:False", action= "store_true")
parser.add_argument('--polyfraction', help="up to this fraction of words per sentence can be polysemous, default: 1.0", type=float, default = 1.0)
parser.add_argument('--test', help = "evaluate on test sentences rather than dev. Default: false.", action = "store_true")
parser.add_argument('--freqmatch', help = "select cloze words only from bin of words with similar frequencies. Default: false", action = "store_true")

args = parser.parse_args()

if args.simlevel not in [-1, 0, 1,2,3]:
    print("Similarity level must be in 0-3, or don't set")
    sys.exit(0)


# settings file
config = configparser.ConfigParser()
config.read("settings.txt")
selpref_method = config["Selpref"]

if args.test:
    testsection = "test"
else:
    testsection = "dev"

print("Testing on section:", testsection)
print("Selectional preferences:", selpref_method["Method"])
print("sim level:", args.simlevel)
print("polysemy fraction:", args.polyfraction)
print("frequency matched cloze words:", args.freqmatch)
          
##########################
# read data
print("Reading data")

vgpath_obj = VGPaths(vgdata = args.vgdata, sdsdata = args.output)

# read frequent objects, attributes, relations
vgcounts_zipfilename, vgcounts_filename = vgpath_obj.vg_counts_zip_and_filename()
with zipfile.ZipFile(vgcounts_zipfilename) as azip:
    with azip.open(vgcounts_filename) as f:
        vgobjects_attr_rel = json.load(f)

# obtain IDs of training and test images

split_zipfilename, split_filename = vgpath_obj.vg_traintest_zip_and_filename()
with zipfile.ZipFile(split_zipfilename) as azip:
    with azip.open(split_filename) as f:
        traintest_split = json.load(f)
    
trainset = set(traintest_split["train"])

vgindex_obj = VgitemIndex(vgobjects_attr_rel)

vgparam_obj = VGParam(vgpath_obj, selpref_method, frequentobj = vgobjects_attr_rel)

global_param, scenario_concept_param, word_concept_param, selpref_param = vgparam_obj.get()

##########################
# determine word frequencies, for the baseline

print("Determining word frequencies")

vgiter = vgiterator.VGIterator()
baseline_counter = Counter()

# count attributes
for img, frequent_it in vgiter.each_image_objects(img_ids = trainset):
    baseline_counter.update([vgindex_obj.o2ix(i) for i in frequent_it])

# count objects
for img, frequent_it in vgiter.each_image_attributes(img_ids = trainset):
    baseline_counter.update([vgindex_obj.a2ix(i) for i in frequent_it])

# count relations
for img, frequent_it in vgiter.each_image_relations(img_ids = trainset):
    baseline_counter.update([vgindex_obj.r2ix(i) for i in frequent_it])


##########################
# randomly select test sentences
print("Selecting and transforming test sentences")

vgsent_obj = VGSentences(vgpath_obj)

# sentences: tuples (sentence ID, word literals, word literals that need to be kept, role literals)
testsentences = [ ]
for sentence in vgsent_obj.each_sentence(vgiter, vgobjects_attr_rel, traintest_split, testsection):
    testsentences.append( sentence)

random.seed(543)
use_testsentences = random.sample(testsentences, args.numsent)

# downsample testsentences now, so that when we select candidates for polysemy later,
# we'll have a better chance of them being adjacent to each other in terms of
# semantic roles
use_testsentences = [ vgsent_obj.transform_sentence_downsample( (sentid, words, [], roles), sentlength_cap = args.maxlen) for sentid, words, roles in use_testsentences]


################################################
# binning
def make_wordbins(wordix_counter, vgindex_obj):
    binstr = "65,100,220,750,10000000"
    bin_ends = [int(b) for b in binstr.split(",")]
    bin_start = 0
    bins = [ ]
    for b in bin_ends:
        bins.append( (bin_start, b) )
        bin_start = b

    print("Bins:", bins)
    wordbins = { }
    for wordtype in [VGOBJECTS, VGATTRIBUTES, VGRELATIONS]:
        # map wordindex, count -> word, count, but keep only the words of the right word type
        words_this_wordtype = [ (vgindex_obj.ix2l(wordix)[0], count) for wordix, count in wordix_counter.items() if vgindex_obj.ix2l(wordix)[1] == wordtype]
        
        wordbins[wordtype] = [ ]
        for b_start, b_end in bins:
            wordbins[wordtype].append( set(w for w, count in words_this_wordtype if count >= b_start and count < b_end))

        # if wordtype == VGATTRIBUTES:
        #     print([(w, c) for w, c in words_this_wordtype if w == "white"], ["white" in b for b in wordbins[wordtype]])

    return wordbins
    
################################################
# transform each test sentence

wordbins = make_wordbins(baseline_counter, vgindex_obj) if args.freqmatch else None
poly_obj = SyntheticPolysemes(vgpath_obj, vgindex_obj, vgobjects_attr_rel, scenario_concept_param, max_fraction = args.polyfraction, wordbins = wordbins)
        
################################################
# now actually transform test sentences
# store gold information for each cloze item
    
# make word IDs for cloze words:
# next word ID is the one after all the objects, attributes, relations so far.
next_wordid = len(vgobjects_attr_rel[VGOBJECTS]) + len(vgobjects_attr_rel[VGATTRIBUTES]) + len(vgobjects_attr_rel[VGRELATIONS])

testsentences_transformed, goldwords = poly_obj.make(use_testsentences, next_wordid, simlevel = args.simlevel, singleword = args.singleword)


gold = { "cloze" : { "clozetype" : "veccloze", "single_word_per_sent?" : args.singleword, "words" : goldwords}}
if args.simlevel > 0:
    gold["cloze"]["simlevel"] = args.simlevel

    
##########################
# write parameters for SDS.
# the only thing we need to change are the concept/word probabilities

print("writing SDS parameters")
    

# record the extra words we got
num_clozewords = len(gold["cloze"]["words"])
global_param["num_words"] += num_clozewords

# # add to word-concept log probabilities:
# # cloze word IDs -> concept ID -> logprob
for word_id, entry in gold["cloze"]["words"].items():

    # word cloze ID -> concept ID -> log of 0.5:
    # equal output probability for both concepts
    word_concept_param[ word_id ] = dict( (entry["concept_ids"][i],  -0.69) for i in [0,1])


vgparam_obj.write(global_param, scenario_concept_param, word_concept_param, selpref_param)



#################
# write cloze sentences

print("Writing test sentences")

vgsent_obj = VGSentences(vgpath_obj)
    
        
###
# write sentences to file.
# no sentence length cap here because we downsampled earlier
vgsent_obj.write(testsentences_transformed, sentlength_cap = None)


###
# write gold information
# print("gold, without baseline info", gold)
gold["baseline"] = dict( baseline_counter)

gold_zipfile, gold_file = vgpath_obj.sds_gold( write = True)
with zipfile.ZipFile(gold_zipfile, "w", zipfile.ZIP_DEFLATED) as azip:
    azip.writestr(gold_file, json.dumps(gold))



