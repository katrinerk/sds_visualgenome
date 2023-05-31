# Katrin Erk January 2023
# Write input for an SDS system:
# parameter files, and input sentences
#
# Here: input for the Cloze test


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
from vgpaths import VGPaths


########3

parser = ArgumentParser()
parser.add_argument('--output', help="directory to write output to, default: sds_in/cloze", default = "sds_in/cloze/")
parser.add_argument('--vgdata', help="directory with VG data including frequent items, train/test split, topic model", default = "data/")
parser.add_argument('--pairs_per_bin', help="Number of cloze pairs to sample per bin, default 20", type = int, default = 20)
parser.add_argument('--numsent', help="Number of sentences per cloze pair, default 50", type = int, default = 50)
parser.add_argument('--bins', help="Bins, given as string of comma-separated nubers, default '65,100,220,750,100000'", default = "65,100,220,750,100000")

args = parser.parse_args()

# settings file
config = configparser.ConfigParser()
config.read("settings.txt")
selpref_method = config["Selpref"]

#############
vgpath_obj = VGPaths(vgdata = args.vgdata, sdsdata = args.output)

# use target words only when they occur
# as arguments of a predicate?
target_needs_to_be_argument = True

# frequency bins from which to sample target words
bin_ends = [int(b) for b in args.bins.split(",")]
bin_start = 0
bins = [ ]
for b in bin_ends:
    bins.append( (bin_start, b) )
    bin_start = b

print("Bins:", bins)

##################
# read data


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


#####3##########3
# obtain training frequencies of objects

vgobj = vgiterator.VGIterator()
obj_count = Counter()

for img, frequent_objects in vgobj.each_image_objects(img_ids = trainset):
    for label in frequent_objects:
        obj_count[ label ] += 1

# # show histogram
# if show_stats:
#     print(text_histogram.histogram(list(obj_count.values())))
#     print("zooming in")
#     print(text_histogram.histogram(list([v for v in obj_count.values() if v < 1700])))
#     print("zooming in again")
#     print(text_histogram.histogram(list([v for v in obj_count.values() if v < 200])))
#     print()

#     for lower, upper in bins:
#         print(lower, "to", upper, sum(1 for v in obj_count.values() if v >= lower and v < upper))

#########
random.seed(6543)

print("Determining cloze word pairs")

#######
# data structure for gold info:
# store under "cloze" for the task, mark "binned" as true
gold = { "cloze" : {"clozetype" : "binned", "words" : { } } }

####
# select object pairs
# pairing words by similar frequency / same bin

# determine bins for objects
bin_objects = { }

for lower, upper in bins:
    bin_objects[ (lower, upper)] = [ w for w, c in obj_count.items() if c >= lower and c < upper]


    
# select object pairs,
# record them in the gold object

# ID of the next word: one past all the objects, attributes, relations we have so far
next_wordid = len(vgobjects_attr_rel[VGOBJECTS]) + len(vgobjects_attr_rel[VGATTRIBUTES]) + len(vgobjects_attr_rel[VGRELATIONS])
# overall number of cloze words
num_clozewords = 0
# mapping concept ID -> cloze word id
conceptid_clozeid = { }
# for each concept, map to its "cloze partner"
clozeconcept_other = { }

testobject_ids = set()

for frequencybin, objects in bin_objects.items():
    # randomly select objects
    if len(objects) < 2 * args.pairs_per_bin:
        print("too few objects in bin", frequencybin, "choosing all", len(objects))
        chosen_objects = objects
    else:
        chosen_objects = random.sample(objects, k= 2* args.pairs_per_bin)

    # make object pairs,
    # record in gold
    offset = int(len(chosen_objects) / 2)
    for i in range(offset):
        obj1 = chosen_objects[i]
        obj2= chosen_objects[i + offset]

        # determine pair, and pair ID, and word, and word ID
        pair = sorted([obj1, obj2])
        pair_ids =  [ vgindex_obj.o2ix(o) for o in pair]
        word_id = next_wordid
        for conceptid in pair_ids:
            conceptid_clozeid[ conceptid ] = word_id
            testobject_ids.add(conceptid)

        clozeconcept_other[obj1] = obj2
        clozeconcept_other[obj2] = obj1
            
        # object frequencies
        pair_freq = [ obj_count[o] for o in pair]
        most_frequent_id = pair_ids[0] if pair_freq[0] >= pair_freq[1] else pair_ids[1]

        
        gold["cloze"]["words"][word_id] = {"concepts" : pair_ids,
                                           "labels" : [obj1, obj2],
                                           "freq" : pair_freq,
                                           "baseline_id" : most_frequent_id,
                                           "word" :  "_".join(pair),
                                           "word_id" : next_wordid,
                                           "bin" : frequencybin,
                                           "occurrences" : 0
                                        }
        
        # keep track of what the ID will be for the next word
        next_wordid += 1
        # and of how many words we've added
        num_clozewords += 1

# for word_id, entry in gold["cloze"]["words"].items():
#     print(word_id, "\n", json.dumps(entry, indent = 2))

##########################
# write parameters for SDS.
# the only thing we need to change are the concept/word probabilities

print("writing SDS parameters")
    
vgparam_obj = VGParam(vgpath_obj, selpref_method)

global_param, scenario_concept_param, word_concept_param, selpref_param = vgparam_obj.get()

# record the extra words we got
global_param["num_words"] += num_clozewords

# add to word-concept log probabilities:
# cloze word IDs -> concept ID -> logprob
for word_id, entry in gold["cloze"]["words"].items():

    # word cloze ID -> concept ID -> log of (1.0)
    # because each underlying concept has a probability of 1 of
    # outputting the cloze word
    word_concept_param[ word_id ] = dict( (entry["concepts"][i],  0.0) for i in [0,1])


vgparam_obj.write(global_param, scenario_concept_param, word_concept_param, selpref_param)

########################
# write sentences for SDS:
# retain only test sentences that contain words of interest,
# optionally: only test sentences that contain words of interest
# as arguments of at least one predicate.
#
# transform test sentences: change the target word to the cloze word.
#
# protect occurrences of the target cloze word, and of
# all predicates that take it as an argument, from
# random downsampling
print("Writing sentences")

vgsent_obj = VGSentences(vgpath_obj)

# store all cloze sentences,
# so we can randomly select among them later
clozeid_sent = defaultdict(list)


for sentid, words, roles in vgsent_obj.each_sentence(vgobj, vgobjects_attr_rel, traintest_split, "test"):
    # sort words into target words to be replaced by cloze, and others.
    # remove target words that aren't arguments if we are doing that
    target_words = []
    nontarget_words = [ ]
    
    for w, objectid, dref in words:

        # target word
        if objectid in testobject_ids:
            if target_needs_to_be_argument:
                if any(dref == argdref for _, _, _, argdref in roles):
                    # ... that is an argument
                    target_words.append( (w, objectid, dref) )
                else:
                    # ... that is not an argument:
                    # don't use
                    pass
            else:
                # ... we're keeping all target words
                target_words.append( (w, objectid, dref) )
        # not a target word
        else:
            nontarget_words.append( (w, objectid, dref))

    # does this sentence contain any of the objects we're manipulating?
    # if not, discard
    if len(target_words) == 0:
        continue

    # does this sentence contain both targets out of a cloze pair?
    # if so, discard the sentence
    #
    # actually, we don't need to do that anymore now that
    # all cloze words have a syntactic context
    # 
    # testobject_ids_this_sent = set(oid for _, oid, _ in target_words)
    # if any(clozeword_partner[oid] in testobject_ids_this_sent for oid in testobject_ids_this_senet):
    #     print("sentence", sentid, "contains both members of a cloze pair, skipping.")
    #    continue

    # this sentence is usable, record as possible test sentence.
    clozeids_this_sent = set(conceptid_clozeid[oid] for _, oid, _ in target_words)
    for clozeid in clozeids_this_sent:
        clozeid_sent[ clozeid ].append( (sentid, target_words, nontarget_words, roles) )

###
# possibly downsample test sentences
# also reshape into dictionary
# sentence ID -> sentence content, relevant cloze pairs
sentid_sent = { }
sentid_clozeids = defaultdict(list)

for clozeid, sentences in clozeid_sent.items():
    
    if args.numsent is not None:
        # we are downsampling
        if len(sentences) > args.numsent:
            sentences = random.sample(sentences, args.numsent)

    # store how many occurrences of the cloze word we have
    gold["cloze"]["words"][clozeid]["occurrences"] = len(sentences)
    
    # store under sentence ID
    for sentid, twords, nontwords, roles in sentences:
        sentid_sent[sentid] = (twords, nontwords, roles)
        sentid_clozeids[sentid].append(clozeid)
                                       


###
# now actually transform sentences
sentences= [ ]
# sentence ID -> cloze IDs and gold concept IDs
gold["cloze"]["sent"] = { }

for sentid, sentence in sentid_sent.items():
    target_words, nontarget_words, roles = sentence

    # transform literals that contain a target
    # to have its cloze word instead.
    # store in "keep words" as we may be downsampling
    # the sentence length later
    keep_wordliterals = [ ]
    wordliterals = [ ]
    gold["cloze"]["sent"][sentid] = [ ]

    cloze_drefs = set()

    for word, oid, dref in target_words:
        # we need to replace this word ID with a word ID for the cloze word
        clozeword_id = conceptid_clozeid[oid]

        # we may or may not be evaluating on this cloze word. are we?
        # if so, record the gold label
        if clozeword_id in sentid_clozeids[sentid]:
            # yes, we are evaluating on this cloze word
            keep_wordliterals.append( (w, clozeword_id, dref) )
            gold["cloze"]["sent"][sentid].append( [clozeword_id, oid] )
            cloze_drefs.add(dref)
            
        else:
            # we are not evaluating on this cloze word.
            # transform anyway, but mark this literal
            # as potentially available for downsampling
            wordliterals.append( (w, clozeword_id, dref) )

    # determine discourse referents of predicates
    # whose arguments are cloze words
    preds_of_cloze_drefs = set(headdref for _, _, headdref, argdref in roles if argdref in cloze_drefs)

    # now sort through other word literals.
    # mark as kept all predicates whose arguments are cloze words,
    # or arguments whose predicates are kepts because they have a cloze argument
    for w, oid, dref in nontarget_words:
        if dref in preds_of_cloze_drefs or any(headdref in preds_of_cloze_drefs for _, _, headdref, argdref in roles if argdref == dref):
            keep_wordliterals.append( (w, oid, dref))
        else:
            wordliterals.append( (w, oid, dref) )
            
    
    sentences.append( [sentid, wordliterals, keep_wordliterals, roles])

        
###
# write sentences to file
vgsent_obj.write(sentences)

# for word_id, entry in gold["cloze"]["words"].items():
#     print(word_id, "\n", json.dumps(entry, indent = 2))

###
# write gold information
gold_zipfile, gold_file = vgpath_obj.sds_gold( write = True)
with zipfile.ZipFile(gold_zipfile, "w", zipfile.ZIP_DEFLATED) as azip:
    azip.writestr(gold_file, json.dumps(gold))

# output info on cloze words to screen
for clozeinfo in gold["cloze"]["words"].values():
    print(clozeinfo["labels"][0], ", ", clozeinfo["labels"][1], " num sentences", clozeinfo["occurrences"])
