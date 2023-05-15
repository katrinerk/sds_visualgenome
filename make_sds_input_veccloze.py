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
parser.add_argument('--numsent', help="Number of sentences to use, default 2000", type = int, default = 2000)
parser.add_argument('--selpref_relfreq', help="selectional preferences using relative frequency rather than similarity to centroid?  default: False", action = "store_true")
parser.add_argument('--maxlen', help="maximum sentence length, default = 25", type = int, default = 25)
parser.add_argument('--simlevel', help="choose cloze pairs only in this similarity range: 0-3, 0 is most similar, default: no restriction", type =int, default = -1)
parser.add_argument('--singleword', help="only one ambiguous word per sentence. default:False", action= "store_true")

args = parser.parse_args()

if args.simlevel not in [-1, 0, 1,2,3]:
    print("Similarity level must be in 0-3, or don't set")
    sys.exit(0)

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
for sentence in vgsent_obj.each_testsentence(vgiter, vgobjects_attr_rel, traintest_split):
    testsentences.append( sentence)

use_testsentences = random.sample(testsentences, args.numsent)

################################################
# transform each test sentence

simlevels = { 0 : [0, 14], 1:[15, 29], 2:[30, 44], 3:[45, 1000000]}

# count how often each argument appears in the pred/arg entries in the vectors
# so that we can later discard pred/arg pairs where the argument only ever
# appears once (because then we'll never find a pair pred'/arg with the same arg
# but different pred')
arg0counter = Counter([a for _, a in vec_obj.predarg0_vec.keys()])
arg1counter = Counter([a for _, a in vec_obj.predarg1_vec.keys()])


##
# make list of candidates for cloze.
# objects, attributes: use as is.
# relations: make two curried variants, with each argument
def candidate_words(words, roles, vgix_obj, frequent_labels, vec_obj, scenario_concept_param, arg0counter, arg1counter):
    ###
    # store mappings dref-> object labels
    dref_obj = defaultdict(list)
    for _, conceptid, dref in words:
        conceptlabel, ctype = vgix_obj.ix2l(conceptid)
        if ctype == "obj":
            dref_obj[dref].append(conceptlabel)

    ###
    # store mapping head dref -> arg dref separately for arg0, arg1
    arg0_drefh_drefd = { }
    arg1_drefh_drefd = { }
    for _, arglabel, dref_h, dref_d in roles:
        if arglabel == "arg0":
            arg0_drefh_drefd[dref_h] = dref_d
        elif arglabel == "arg1":
            arg1_drefh_drefd[dref_h] = dref_d
        else:
            raise Exception("unknown role label " + str(arglabel))


    ###
    # determine candidates
    # retv: list of tuples (candidate type, candidate info, word index)
    retv = [ ]
    for wordix, wordentry in enumerate(words):
        _, conceptid, dref = wordentry
        conceptlabel, ctype = vgix_obj.ix2l(conceptid)
    
        if ctype == "obj":
            # object: no currying necessary. but is this frequent enough?
            if conceptlabel in  frequent_labels["objects"] and conceptlabel in vec_obj.object_vec.keys() and conceptid in scenario_concept_param:
                # yes, usable
                retv.append( ("object",conceptlabel, wordix) )

        elif ctype == "att":
            # attribute: no currying necessary. but is this frequent enough?
            if conceptlabel in  frequent_labels["attributes"] and conceptlabel in vec_obj.attrib_vec.keys() and conceptid in scenario_concept_param:
                # yes, usable
                retv.append( ( "attr", conceptlabel, wordix) )
                
        else:
            # relation: we need to curry this, if it's frequent enough
            if conceptlabel in frequent_labels["relations"] and conceptid in scenario_concept_param:
                # may be usable if the argument is frequent enough
                if dref not in arg0_drefh_drefd or dref not in arg1_drefh_drefd:
                    # no arguments recorded, skip
                    continue

                # retain only objects that have a vector together with the predicate
                # and that appear with at least 2 predicates in the vectors list
                predarg0s = [ ("predarg0", conceptlabel, alabel) for alabel in dref_obj[ arg0_drefh_drefd[ dref ]] \
                                  if (conceptlabel, alabel) in  vec_obj.predarg0_vec.keys() and arg0counter[alabel] >= 10]
                predarg1s = [ ("predarg1", conceptlabel, alabel) for alabel in dref_obj[ arg1_drefh_drefd[ dref]] \
                                  if (conceptlabel, alabel) in vec_obj.predarg1_vec.keys() and arg1counter[alabel] >= 10]


                if len(predarg0s) > 0 or len(predarg1s) > 0:
                    retv.append( (ctype, predarg0s + predarg1s, wordix) )

    return retv
                    

#############3
def sample_clozepair(wordtype, wordinfo, simlevel,simlevel_vals, vec_obj, vgix_obj):
    
    # determine ranked similarities
    if wordtype == "object":
        # object: wordinfo is simply a concept label

        word1 = wordinfo

        # compute similarity based on embeddings
        neighbors = [n for n, _ in vec_obj.ranked_sims(word1, wordtype)[1:] if vgix_obj.isobj(n)]

        # sample a 2nd word in simlevel, return its index and the percentile of the index
        word2index, word2rank = sample_cloze_fromlist(neighbors, simlevel, simlevel_vals)
        word2 = neighbors[word2index]

        # map word to word ID
        word1id = vgix_obj.o2ix(word1) 
        word2id = vgix_obj.o2ix(word2)

    elif wordtype == "attr":
        # object: wordinfo is simply a concept label

        word1 = wordinfo

        # compute similarity based on embeddings
        neighbors = [n for n, _ in vec_obj.ranked_sims(word1, wordtype)[1:] if vgix_obj.isatt(n)]

        # sample a 2nd word in simlevel, return its index and the percentile of the index
        word2index, word2rank = sample_cloze_fromlist(neighbors, simlevel, simlevel_vals)
        word2 = neighbors[word2index]

        # map word to word ID
        word1id = vgix_obj.a2ix(word1) 
        word2id = vgix_obj.a2ix(word2)

    elif wordtype == "rel":
        # relation: wordtype is a list of triples (predarg0/1, predlabel, arglabel)
        # sample one of the triples, determine neighbors
        argtype, pred1, arg1 = random.choice(wordinfo)
        word1 = pred1

        # filter neighbors: have to have different pred, same arg
        neighbors = [n[0] for n, _ in vec_obj.ranked_sims((pred1, arg1), argtype)[1:] if vgix_obj.isrel(n[0]) and n[0] != pred1 and n[1] == arg1]
        if len(neighbors) == 0:
            # print("No neighbors for pred/arg pair, skipping:", pred1, arg1, argtype)
            return None

        # sample a 2nd word in simlevel, return its index and the percentile of the index        
        word2index, word2rank = sample_cloze_fromlist(neighbors, simlevel, simlevel_vals)
        
        # what is the actual 2nd word? We had transformed neighbors to only retain pred before
        word2 = neighbors[word2index]

        # map word to word ID
        word1id = vgix_obj.r2ix(word1)
        word2id = vgix_obj.r2ix(word2)

    else:
        raise Exception( "unknown wordtype " + str(wordtype))


    if word1id is None:
        print("could not determine ID for", word1, wordtype)
        return None
    elif word2id is None:
        print("could not determine ID for", word2, wordtype)
        return None

    return (word1, word1id, word2, word2id, word2rank)
    

def sample_cloze_fromlist(neighbors, simlevel, simlevel_vals):
    if simlevel < 0:
        # simlevel not set?
        # choose one of the simlevels at random
        simlevel = random.choice(list(simlevel_vals.keys()))

    simlevel_lower, simlevel_upper = simlevel_vals[ simlevel]
    simlevel_upper = min([simlevel_upper, len(neighbors) -1])

    # select a neighbor, either from anywhere or from the selected level
    if len(neighbors) < 45:
        # sample from anywhere, we have too little data to distinguish levels
        index = random.randint(0, len(neighbors)-1)

    else:
        index = random.randint(simlevel_lower, simlevel_upper)


    word2rank = index / len(neighbors)

    return (index, word2rank)
    
        
################################################
# now actually transform test sentences
# store gold information for each cloze item
gold = { "cloze" : { "clozetype" : "veccloze", "single_word_per_sent?" : args.singleword, "words" : { }}}
if args.simlevel > 0:
    gold["cloze"]["simlevel"] = args.simlevel

    
# make word IDs for cloze words:
# next word ID is the one after all the objects, attributes, relations so far.
next_wordid = len(vgobjects_attr_rel["objects"]) + len(vgobjects_attr_rel["attributes"]) + len(vgobjects_attr_rel["relations"])

testsentences_transformed = [ ]

ctype_map = {"object" : "obj", "attr" : "att", "predarg0" : "rel", "predarg1" : "rel", "rel" : "rel"}

for sentid, words, roles in use_testsentences:
    
    # print("---Sentence:-------")
    # for _, cid, dref in words:
    #     print(vgindex_obj.ix2l(cid)[0] + "(" + str(dref) + ")", end = ", ")
    # print()
    # for _, arglabel, dref_h, dref_d in roles:
    #     print(arglabel+ "(" + str(dref_h) + "," + str(dref_d) + ")", end = ", ")
    # print("\n--")
    
    ##
    # determine words that could be made into cloze items.
    # for roles, use curried words
    candidates = candidate_words(words, roles, vgindex_obj, vgobjects_attr_rel, vec_obj, scenario_concept_param, arg0counter, arg1counter)
    # print("Candidates")
    # for ctype, cdata, wordix in candidates:
    #     print(str(wordix) + ":" + str(cdata), end = ", ")
    # print()

    ##
    # select candidates to make cloze items: if args.singleword, then one, 
    # else up to 1/2 of all words in the sentence
    numwords_for_cloze = 1 if args.singleword else random.randint(1, int(len(words) / 2))
    words_for_cloze = random.sample(candidates, numwords_for_cloze)
    # print("Selected")
    # for ctype, cdata, wordix in words_for_cloze:
    #     print(str(wordix) + ":" + str(cdata) + "/" + ctype, end = ", ")
    # print()

    ##
    # make cloze words, store in gold, store transformed literal

    targetwords = [ ]
    targetword_ids = [ ]
    
    for ctype, cdata, ix in words_for_cloze:
        # find second word as cloze partner for the word in cdata.
        # simlevel says whether to restrict similarities to a particular quartile in the ranked list
        # of vector neighbors for word1
        retv = sample_clozepair(ctype, cdata, args.simlevel, simlevels, vec_obj, vgindex_obj)
        if retv is None:
            # print("Unable to sample cloze pair for", cdata, ctype, ix)
            continue
        word1, word1id, word2, word2id, rel_rank_of_word2 = retv
        # print("cloze pair:", vgindex_obj.ix2l(word1id)[0], vgindex_obj.ix2l(word2id)[0], rel_rank_of_word2, next_wordid)

        # types recorded here: obj, att, rel
        
        gold["cloze"]["words"][next_wordid] = {
            "concept_ids" : [word1id, word2id],
            "gold_id" : word1id,
            "ctype" : ctype_map[ctype],
            "word" : str(word1) + "_" + str(word2)
            }

        # make transformed target word
        w, _, dref = words[ix]
        targetwords.append( [w, next_wordid, dref] )
        # keep indices of actual targetwords (ones for which we didn't get a None above)
        targetword_ids.append(ix)
        
        next_wordid += 1

    if len(targetwords) == 0:
        # something went wrong, and we didn't successfully choose target words for this sentence
        print("No target words successfully selected for sentence, skipping", sentid)
        continue

    ##
    # what are the non-target words?
    otherwords = [w for i, w in enumerate(words) if i not in targetword_ids]

    # print("targetwords", targetwords)
    # print("other", [(vgindex_obj.ix2l(w)[0], d) for _, w, d in otherwords])
    
    ## transformed testsentence is done
    testsentences_transformed.append( [sentid, otherwords, targetwords, roles] )

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
# write sentences to file
vgsent_obj.write(testsentences_transformed, sentlength_cap = args.maxlen)


###
# write gold information
# print("gold, without baseline info", gold)
gold["baseline"] = dict( baseline_counter)

gold_zipfile, gold_file = vgpath_obj.sds_gold( write = True)
with zipfile.ZipFile(gold_zipfile, "w", zipfile.ZIP_DEFLATED) as azip:
    azip.writestr(gold_file, json.dumps(gold))



