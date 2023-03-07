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
import gensim

import vgiterator
from sds_input_util import VGSentences, VGParam
from vgindex import VgitemIndex
from vec_util import VectorInterface

from vgpaths import VGPaths


########3

parser = ArgumentParser()
parser.add_argument('--output', help="directory to write output to, default: sds_in/veccloze", default = "sds_in/veccloze/")
parser.add_argument('--vgdata', help="directory with VG data including frequent items, train/test split, topic model", default = "data/")
parser.add_argument('--pairs_per_cond', help="Number of cloze pairs to sample per condition, default 40", type = int, default = 5)
parser.add_argument('--numsent', help="Number of sentences per cloze pair, default 50", type = int, default = 50)
parser.add_argument('--top_n_sim', help="Number of top n neighbors from which to select a cloze pair, default 10", type = int, default = 10)
parser.add_argument('--selpref_relfreq', help="selectional preferences using relative frequency rather than similarity to centroid?  default: False", action = "store_true")
parser.add_argument('--scen_per_concept', help="Number of top scenarios to record for a concept, default 5", type = int, default = 5)

args = parser.parse_args()

##########################
# read data

vgpath_obj = VGPaths(vgdata = args.vgdata, sdsdata = args.output)

vec_obj = VectorInterface(vgpath_obj)

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

random.seed(543)

vgindex_obj = VgitemIndex(vgobjects_attr_rel)

vgparam_obj = VGParam(vgpath_obj, top_scenarios_per_concept = args.scen_per_concept, selpref_vectors = not(args.selpref_relfreq),
                      frequentobj = vgobjects_attr_rel)

global_param, scenario_concept_param, word_concept_param, selpref_param = vgparam_obj.get()

##########################
# determine word frequencies, for the baseline


vgobj = vgiterator.VGIterator()
attr_count = Counter()

for img, frequent_it in vgobj.each_image_attributes(img_ids = trainset):
    for label in frequent_it:
        attr_count[ label ] += 1

##########################
#
# usable labels:
# need to be frequent, AND have a vector, AND have scenarios
def usable_labels(labeltype, frequent_labels, vgindex_obj, vec_obj, scenario_concept_param):
    if labeltype == "attr":
        # attribute label that is frequent, has a vector and has a scenario
        return [ell for ell in frequent_labels["attributes"] if ell in vec_obj.attrib_vec.keys() and\
                vgindex_obj.a2ix(ell) in scenario_concept_param]
    elif labeltype = "predarg0":
        return [predargl for predargl in vec_obj.predarg0.keys() if predargl[0] in frequent_labels["relations"] and\
                predargl[1] in frequent_labels["objects"] and vgindex_obj.r2ix(predargl[0]) in scenario_concept_param]
    elif labeltype == "predarg1":
        return [predargl for predargl in vec_obj.predarg1.keys() if predargl[0] in frequent_labels["relations"] and\
                predargl[1] in frequent_labels["objects"] and vgindex_obj.r2ix(predargl[0]) in scenario_concept_param]
    else:
        raise Exception("unknown label type " + labeltype)
    

# find second member of cloze pair for each first member
def each_completed_clozepair(labeltype, first_members_of_clozepairs, labels_to_choose_from, top_n_sim):
    # for each first member of a cloze pair, find a second member
    previously_chosen= set(first_members_of_clozepairs)

    # count = 0
    for ell1 in first_members_of_clozepairs:

        # count += 1
        # if count > 10:
        #     break
        # rank all other words of the same category by similarity to ell1
        neighbors = vec_obj.ranked_sims(ell1, labeltype)

        # determine top n neighbors
        top_n = [ ]
        for ell2, sim in neighbors:
            if len(top_n) >= top_n_sim:
                # we have found as many neighbors as we needed
                break
            if ell2 in labels_to_choose_from and ell2 not in previously_chosen:
                top_n.append(ell2)

        if len(top_n) == 0:
            print("Error: No candidates remaining as cloze partner for", ell1)
            continue

        ell2 = random.choice(top_n)
        previously_chosen.add(ell2)

        # print("word", ell1, "nearest neighbors")
        # for l, s in neighbors[:10]: print("\t", l, s)
        # print("top n remaining", ", ".join(top_n))
        # print("chosen:", ell2)
        yield (ell1, ell2)

# given a sentence as VGSentences yields it, "curry" it,
# combining labels of predicates with either first or second arguments
def curry_sentence(sentence, arglabel):
    words, roles = sentence
    
    # store mappings dref-> words and head dref -> dependent dref in the argument relation of interest
    dref_words = defaultdict(list)
    drefh_drefd = { }
    # word labels
    for _, label, dref in words:
        dref_words[ dref].append(label)
    # head/dependent relations
    for _, thisarglabel, dref_h, dref_d in roles:
        # only store for the argument relation of interest
        if thisarglabel == arglabel:
            drefh_drefd[ dref_h] = dref_d

    # now transform word labels
    newwords = [ ]
    for w, label, dref in words:
        if dref in drefh_drefd:
            # this dref indicates a relation or attribute
            # with an argument in the relation of interest
            deplabels = dref_words[ drefh_drefd[ dref]]

            newwords.append( (w, [ (label, dl) for dl in deplabels], dref))
        else:
            newwords.append( (w, label, dref))

    return (newwords, roles)

def wordliteral_label_matches(label_or_labels, conceptids):

    if isinstance(label_or_labels, list):
        return [ell for ell in label_or_labels if ell in conceptids]
    elif isinstance(label_or_labels, int):
        if label_or_labels in conceptids:
            return [label_or_labels]
        else:
            return [ ]
    else:
        return [ ]
    
##########################
# make cloze pairs:

gold = { "cloze" : {"binned" : 0, "clozetype" : "att_rel", "words" : { }}}
                                                                                   
next_wordid = len(vgobjects_attr_rel["objects"]) + len(vgobjects_attr_rel["attributes"]) + len(vgobjects_attr_rel["relations"])
# mapping concept ID -> cloze word id
conceptid_clozeid = { }


for labeltype in ["attr", "predarg0", "predarg1"]:

    # determine labels that can go into a cloze pair
    all_labels = usable_labels("attr", vgobjects_attr_rel, vgindex_obj, vec_obj, scenario_concept_param)
    # select first members of cloze pairs
    first_members_of_clozepairs = random.sample(all_labels, args.pairs_per_cond)

    for ell1, ell2 in each_completed_clozepair(labeltype, first_members_of_clozepairs, all_labels, args.top_n_sim):
        
        pair = sorted([ell1, ell2])
        pair_ids = [vgindex_obj.a2ix(ell) for ell in pair]
        pair_freq = [attr_count[ell] for ell in pair]
        most_frequent_id = pair_ids[0] if pair_freq[0] >= pair_freq[1] else pair_ids[1]
        cloze_id = next_wordid

        # HIER doesn't work for relations,
        # and for cloze pairs that are curried
        gold["cloze"]["words"][cloze_id] = {"concepts" : pair_ids,
                                            "labels" : pair,
                                            "freq" : pair_freq,
                                            "baseline_id" : most_frequent_id,
                                            "word" :  "_".join(pair),
                                            "word_id" : cloze_id,
                                            "type" : labeltype,
                                            "occurrences" : 0
                                            }
        next_wordid += 1
        conceptid_clozeid[ pair_ids[0] ]= cloze_id
        conceptid_clozeid[ pair_ids[1] ]= cloze_id


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

    # word cloze ID -> concept ID -> log of (1.0)
    # because each underlying concept has a probability of 1 of
    # outputting the cloze word
    word_concept_param[ word_id ] = dict( (entry["concepts"][i],  0.0) for i in [0,1])


vgparam_obj.write(global_param, scenario_concept_param, word_concept_param, selpref_param)

###########################3
# transform some test sentences
# retain only test sentences that contain words of interest,
#
# transform test sentences: change the target word to the cloze word.
#
# protect occurrences of the target cloze word, and of
# all predicates that take it as an argument, from
# random downsampling
print("Writing sentences")

vgsent_obj = VGSentences(vgpath_obj)


####33
# HIER
predarg0_ids = [ (vgindex_obj.r2ix(rellabel), vgindex_obj.o2ix(arglabel)) for rellabel, arglabel in vec_obj.predarg0_vec ]
predarg0_counts = Counter()

for sentid, words, roles in vgsent_obj.each_testsentence(vgobj, vgobjects_attr_rel, traintest_split):
    newwords, newroles = curry_sentence((words, roles), "arg0")
    
    for w, label_or_labels, dref in newwords:

        # target word?
        matches =  wordliteral_label_matches(label_or_labels, predarg0_ids)

        for emm in matches:
            predarg0_counts[emm] += 1

#import text_histogram

# text_histogram.histogram(predarg0_counts.values(), custbuckets = "0,10,50,100,1000,2500")
# sys.exit(0)

# store all cloze sentences,
# so we can randomly select among them later
clozeid_sent = defaultdict(list)

for sentid, words, roles in vgsent_obj.each_testsentence(vgobj, vgobjects_attr_rel, traintest_split):
    # sort words into target words to be replaced by cloze, and others.
    target_words = []
    nontarget_words = [ ]
    
    for w, conceptid, dref in words:

        # target word
        if conceptid in conceptid_clozeid:
            target_words.append( (w, conceptid, dref) )
        # not a target word
        else:
            nontarget_words.append( (w, conceptid, dref))

    # does this sentence contain any of the objects we're manipulating?
    # if not, discard
    if len(target_words) == 0:
        continue

    # this sentence is usable, record as possible test sentence.
    clozeids_this_sent = set(conceptid_clozeid[iid] for _, iid, _ in target_words)
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


###
# write gold information
gold_zipfile, gold_file = vgpath_obj.sds_gold( write = True)
with zipfile.ZipFile(gold_zipfile, "w", zipfile.ZIP_DEFLATED) as azip:
    azip.writestr(gold_file, json.dumps(gold))

# output info on cloze words to screen
for clozeinfo in gold["cloze"]["words"].values():
    print(clozeinfo["labels"][0], ", ", clozeinfo["labels"][1], " num sentences", clozeinfo["occurrences"])


#########3
