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
parser.add_argument('--pairs_objattr', help="Number of obj or attribute cloze pairs to sample, default 40", type = int, default = 40)
parser.add_argument('--pairs_predarg', help="Number of pred/arg cloze pairs to sample for each role, default 120", type = int, default = 120)
parser.add_argument('--numsent', help="Number of sentences per attr. cloze pair, default 50", type = int, default = 50)
parser.add_argument('--top_n_sim', help="Number of top n neighbors from which to select a cloze pair, default 10", type = int, default = 10)
parser.add_argument('--selpref_relfreq', help="selectional preferences using relative frequency rather than similarity to centroid?  default: False", action = "store_true")

args = parser.parse_args()

##########################
# read data
print("Reading data")

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

vgparam_obj = VGParam(vgpath_obj, selpref_vectors = not(args.selpref_relfreq),
                      frequentobj = vgobjects_attr_rel)

global_param, scenario_concept_param, word_concept_param, selpref_param = vgparam_obj.get()

##########################
# determine word frequencies, for the baseline

print("Determining word frequencies")

vgiter = vgiterator.VGIterator()
counters = { "object" : Counter(),
             "attr" : Counter(),
             "predarg0" : Counter(),
             "predarg1" : Counter()
             }

# count attributes
for img, frequent_it in vgiter.each_image_attributes(img_ids = trainset):
    counters["attr"].update(frequent_it)


# collect labels of potential arguments
img_objid_label = { }

for imgid, objects in vgiter.each_image_objects_full(img_ids = trainset):
    img_objid_label[imgid] = { }
    for objid, names in objects:
        img_objid_label[imgid][objid] = names
        counters["object"].update(names)

# count pred/arg0, pred/arg1 pairs
for imgid, rel_arg in vgiter.each_image_relations_full(img_ids = trainset):
    for predname, subjid, objid in rel_arg:
        if imgid not in img_objid_label or objid not in img_objid_label[imgid] or subjid not in img_objid_label[imgid]:
            print("Warning: unexpectedly didn't find rel argument", imgid, subjid, objid)
            continue
        
        for l in img_objid_label[imgid][subjid]:
            counters["predarg0"][(predname, l)] += 1
        for l in img_objid_label[imgid][objid]:
            counters["predarg1"][(predname, l)] += 1
        
##########################
#
# usable labels:
# need to be frequent, AND have a vector, AND have scenarios
def usable_labels(labeltype, frequent_labels, vgindex_obj, vec_obj, scenario_concept_param):
    if labeltype == "object":
        return [ell for ell in frequent_labels["objects"] if ell in vec_obj.object_vec.keys()]
    
    elif labeltype == "attr":
        # attribute label that is frequent, has a vector and has a scenario
        return [ell for ell in frequent_labels["attributes"] if ell in vec_obj.attrib_vec.keys() and\
                vgindex_obj.a2ix(ell) in scenario_concept_param]
                
    elif labeltype == "predarg0":
        return [predargl for predargl in vec_obj.predarg0_vec.keys() if predargl[0] in frequent_labels["relations"] and\
                predargl[1] in frequent_labels["objects"] and vgindex_obj.r2ix(predargl[0]) in scenario_concept_param]
                
    elif labeltype == "predarg1":
        return [predargl for predargl in vec_obj.predarg1_vec.keys() if predargl[0] in frequent_labels["relations"] and\
                predargl[1] in frequent_labels["objects"] and vgindex_obj.r2ix(predargl[0]) in scenario_concept_param]
    else:
        raise Exception("unknown label type " + labeltype)
    

# find second member of cloze pair for each first member
def each_completed_clozepair(labeltype, first_members_of_clozepairs, labels_to_choose_from, top_n_sim, remove_same_pred= False):
    # for each first member of a cloze pair, find a second member
    previously_chosen= set(first_members_of_clozepairs)

    for ell1 in first_members_of_clozepairs:

        # rank all other words of the same category by similarity to ell1
        neighbors = vec_obj.ranked_sims(ell1, labeltype)

        # determine top n neighbors
        top_n = [ ]
        for ell2, sim in neighbors:
            if len(top_n) >= top_n_sim:
                # we have found as many neighbors as we needed
                break
            if remove_same_pred and ell2[0] == ell1[0]:
                # labels are two-part, predicate argument.
                # we are not considering labels that have the same predicate
                # as all1
                continue
            
            if ell2 in labels_to_choose_from and ell2 not in previously_chosen:
                # good label
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

    
##########################
# make cloze pairs:

print("Making cloze pairs")

gold = { "cloze" : {"binned" : 0, "clozetype" : "vecbased", "words" : { }}}
                                                                                   
next_wordid = len(vgobjects_attr_rel["objects"]) + len(vgobjects_attr_rel["attributes"]) + len(vgobjects_attr_rel["relations"])
# mapping concept ID -> cloze word id
conceptid_clozeid = { "object" : { }, "attr" : {}, "predarg0": { }, "predarg1" : { }}


for labeltype, num_pairs_to_choose in [("object", args.pairs_objattr), ("attr", args.pairs_objattr), ("predarg0", args.pairs_predarg), ("predarg1", args.pairs_predarg)]:

    # determine labels that can go into a cloze pair
    all_labels = usable_labels(labeltype, vgobjects_attr_rel, vgindex_obj, vec_obj, scenario_concept_param)
    # select first members of cloze pairs
    first_members_of_clozepairs = random.sample(all_labels, num_pairs_to_choose)

    for ell1, ell2 in each_completed_clozepair(labeltype, first_members_of_clozepairs, all_labels, args.top_n_sim,
                                               remove_same_pred = labeltype.startswith("pred")):
        
        pair = sorted([ell1, ell2])

        if labeltype == "object":
            pair_ids = [vgindex_obj.o2ix(ell) for ell in pair]
            concept_ids = pair_ids
            pair_freq = [counters["object"][ell] for ell in pair]
            clozeword = "|".join(pair)
            
        elif labeltype == "attr":
            pair_ids = [vgindex_obj.a2ix(ell) for ell in pair]
            concept_ids = pair_ids
            pair_freq = [counters["attr"][ell] for ell in pair]
            clozeword = "|".join(pair)
            
        else:
            pair_ids = [(vgindex_obj.r2ix(ell[0]), vgindex_obj.o2ix(ell[1])) for ell in pair]
            concept_ids = [ p for p, a in pair_ids]
            pair_freq = [counters[labeltype][ell] for ell in pair] 
            clozeword = "|".join(["_".join(m) for m in pair]).replace(" ", "_")
            
        most_frequent_id = pair_ids[0] if pair_freq[0] >= pair_freq[1] else pair_ids[1]
        cloze_id = next_wordid

        gold["cloze"]["words"][cloze_id] = {"concepts" : concept_ids,
                                            "labels" : pair,
                                            "freq" : pair_freq,
                                            "baseline_id" : most_frequent_id,
                                            "word" :  clozeword,
                                            "word_id" : cloze_id,
                                            "type" : labeltype,
                                            "occurrences" : 0
                                            }
        next_wordid += 1
        conceptid_clozeid[labeltype][ pair_ids[0] ]= cloze_id
        conceptid_clozeid[labeltype][ pair_ids[1] ]= cloze_id

# for clozeid, entry in gold["cloze"]["words"].items():
#     print(clozeid, "word", entry["word"], "labels", entry["labels"], "label IDs", entry["concepts"], "frequencies", entry["freq"],
#               "baseline", entry["baseline_id"])

# sys.exit(0)

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
print("Determining sentences with cloze pairs")

# given a sentence as VGSentences yields it, "curry" it,
# combining labels of predicates with either first or second arguments
def curry_sentence(sentence, arglabel):
    words, roles = sentence
    
    # store mappings dref-> words and head dref -> dependent dref in the argument relation of interest
    dref_words = defaultdict(list)
    drefh_drefd = { }
    # word labels:
    # map discourse referent to set of word labels
    for _, label, dref in words:
        dref_words[ dref].append(label)
    # head/dependent relations
    # map each head to at most one dependent that it has
    # in the role of interest (arglabel)
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

            for dl in deplabels:
                newwords.append( (w, (label, dl), dref) )

        else:
            newwords.append( (w, label, dref))

    return (newwords, roles)

# from curried word literal, remove argument label:
# assume shape ("w", labelpair, dref)
# where labelpair has  the form (predlabel, arglabel):
# return ("w", predlabel, dref)
def uncurry_word(word):
    w, labelpair, dref = word

    return (w, labelpair[0], dref)

###

vgsent_obj = VGSentences(vgpath_obj)


# store all cloze sentences,
# so we can randomly select among them later
clozeid_sent = defaultdict(list)

for sentid, words, roles in vgsent_obj.each_testsentence(vgiter, vgobjects_attr_rel, traintest_split):
    # sort words into target words to be replaced by cloze, and others.
    # target words: mapping from literals to cloze IDs
    target_words = defaultdict(list)
    clozeids_this_sent = set()
    

    # one pass over the words without currying, for objects and attributes    
    for w, conceptid, dref in words:
        # target word
        for labeltype in ["object", "attr"]:
            if conceptid in conceptid_clozeid[labeltype]:
                target_words[(w, conceptid, dref) ].append(conceptid_clozeid[labeltype][ conceptid] )
                clozeids_this_sent.add(conceptid_clozeid[labeltype][ conceptid])


    # one pass for predicate/arg0, one for arg1
    for labeltype in ["predarg0", "predarg1"]:
        role = labeltype[4:]
        # integrate arguments into predicate labels
        currywords, _ =  curry_sentence((words, roles), role)
        
        for w, conceptidpair, dref in currywords:
            # if this concept ID has a cloze ID for this label type
            if conceptidpair in conceptid_clozeid[labeltype]:
                # store the un-curried word as the target word
                target_words[uncurry_word( (w, conceptidpair, dref) ) ].append(conceptid_clozeid[labeltype][ conceptidpair ])
                clozeids_this_sent.add(conceptid_clozeid[labeltype][ conceptidpair])

    # does this sentence contain any of the objects we're manipulating?
    # if not, discard
    if len(clozeids_this_sent) == 0:
        continue

    # this sentence is usable, record as possible test sentence.

    # determine non-target words
    nontarget_words = [ w for w in words if tuple(w) not in target_words.keys()]

    # and record
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



print("Transforming and writing test sentences")

###
# now actually transform sentences
sentences= [ ]
# sentence ID -> cloze IDs and gold concept IDs
gold["cloze"]["sent"] = { }

for sentid, sentence in sentid_sent.items():
    targetwords_clozeid_dict, nontarget_words, roles = sentence

    # transform literals that contain a target
    # to have its cloze word instead.
    # store in "keep words" as we may be downsampling
    # the sentence length later
    keep_wordliterals = [ ]
    wordliterals = nontarget_words.copy()
    gold["cloze"]["sent"][sentid] = [ ]

    for literal, clozeids in targetwords_clozeid_dict.items():
        w, oid, dref = literal

        for clozeword_id in clozeids:
            if clozeword_id in sentid_clozeids[sentid]:
                # we are actually evaluating on this cloze word
                keep_wordliterals.append( (w, clozeword_id, dref) )
                gold["cloze"]["sent"][sentid].append( [clozeword_id, oid] )
            else:
                # we are not evaluating on this cloze word.
                # transform anyway, but mark this literal
                # as potentially available for downsampling
                wordliterals.append( (w, clozeword_id, dref) )
    
    sentences.append( [sentid, wordliterals, keep_wordliterals, roles])


# for sentid, wordliterals, keep_wordliterals, roles in sentences:
#     print("---------------")
#     print("HIER", sentid)
#     for _, conceptid, dref in keep_wordliterals:
#         print("target", conceptid, dref)
#     print("__Roles__")
#     for _, pred, drefh, drefd in roles:
#         print("role", pred, drefh, drefd)


    
        
###
# write sentences to file
vgsent_obj.write(sentences, sentlength_cap = None)


###
# write gold information
gold_zipfile, gold_file = vgpath_obj.sds_gold( write = True)
with zipfile.ZipFile(gold_zipfile, "w", zipfile.ZIP_DEFLATED) as azip:
    azip.writestr(gold_file, json.dumps(gold))

# output info on cloze words to screen
for clozeinfo in gold["cloze"]["words"].values():
    print(clozeinfo["labels"][0], ", ", clozeinfo["labels"][1], " num sentences", clozeinfo["occurrences"])


