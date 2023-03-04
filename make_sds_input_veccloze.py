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
from vgpaths import VGPaths


########3

parser = ArgumentParser()
parser.add_argument('--output', help="directory to write output to, default: sds_in/cloze", default = "sds_in/cloze/")
parser.add_argument('--vgdata', help="directory with VG data including frequent items, train/test split, topic model", default = "data/")
parser.add_argument('--pairs_per_cond', help="Number of cloze pairs to sample per condition, default 40", type = int, default = 5)
parser.add_argument('--sents_per_pair', help="Number of sentences per cloze pair, default 50", type = int, default = 50)
parser.add_argument('--top_n_sim', help="Number of top n neighbors from which to select a cloze pair, default 20", type = int, default = 20)
parser.add_argument('--scen_per_concept', help="Number of top scenarios to record for a concept, default 5", type = int, default = 5)

args = parser.parse_args()


vgpath_obj = VGPaths(vgdata = args.vgdata, sdsdata = args.output)

vec_obj = VectorInterface(vgpath_obj)

##########################
# select cloze pairs:
# attributes


# read frequent objects, attributes, relations
vgcounts_zipfilename, vgcounts_filename = vgpath_obj.vg_counts_zip_and_filename()
with zipfile.ZipFile(vgcounts_zipfilename) as azip:
    with azip.open(vgcounts_filename) as f:
        vgobjects_attr_rel = json.load(f)


random.seed(543)

# select args.pairs_per_cond attributes
first_members_of_clozepairs = random.choice(vgobjects_attr_rel["attributes"], args.pairs_per_cond)

# for each first member of a cloze pair, find a second member
previously_chosen_attr= set(first_members_of_clozepairs)

attr_clozepairs = [ ]
for attr1 in first_members_of_clozepairs:
    neighbors = vec_obj.ranked_sims(attr1, "attr")
    
    top_n = [ ]
    num_selected = 0
    for itemindex, sim in neighbors:
        if num_selected >= args.top_n_sim:
                # we have found as many neighbors as we needed
                break
        if itemindex not in previously_chosen_attr:
            top_n.append(itemindex)
            previously_chosen_attr.add(itemindex)
            num_selected += 1

    if len(top_n) == 0:
        print("Error: Did not find any remaining cloze partner for ", attr1)
        continue

    attr2 = random.choice(top_n)

    attr_clozepairs.append( (attr1, attr2))
    
##########################
# write parameters for SDS.
# the only thing we need to change are the concept/word probabilities

print("writing SDS parameters")
    
vgparam_obj = VGParam(vgpath_obj, top_scenarios_per_concept = args.scen_per_concept, selpref_vectors = not(args.selpref_relfreq))

global_param, scenario_concept_param, word_concept_param, selpref_param = vgparam_obj.get()

# record the extra words we got
# global_param["num_words"] += num_clozewords

# # add to word-concept log probabilities:
# # cloze word IDs -> concept ID -> logprob
# for word_id, entry in gold["cloze"]["words"].items():

#     # word cloze ID -> concept ID -> log of (1.0)
#     # because each underlying concept has a probability of 1 of
#     # outputting the cloze word
#     word_concept_param[ word_id ] = dict( (entry["concepts"][i],  0.0) for i in [0,1])


vgparam_obj.write(global_param, scenario_concept_param, word_concept_param, selpref_param)

# vecfilename = vgpath_obj.vg_vecfilename()
# object_vec = { }
# attrib_vec = { }
# predarg0_vec = { }
# predarg1_vec = { }

# ###
# # read vectors,
# # vectors for objects will be available in
# # object_vec, a dictionary
# # object label -> vector of floating point numbers
# with open(vecfilename) as f:
#     for line in f:
#         pieces = line.split()
#         # first word is label
#         label = pieces[0]
#         # from what I can see, vectors for attributes have no .n at the end,
#         # and vectors for relations are actually for relation-argument pairs,
#         # and always have parentheses.
#         # what are they for?
#         #
#         # determine vectors for objects:
#         # remove .n
#         # and turn underscores to spaces
#         if label.endswith(".n") and "(" not in label and ")" not in label:
#             label = label[:-2]
#             label = label.replace("_", " ")
#             object_vec[ label ] = np.array([ float(v) for v in pieces[1:]])
#         elif "(" in label and ")" in label:
#             match = re.match("(.+)\((.+),(.+)\)$", label)
#             if match is None:
#                 print("could not decompose label", label)
#                 continue

#             predicate = match[1]
#             arg0 = match[2]
#             arg1 = match[3]

#             predicate = predicate.replace("|", " ")
#             if arg0 == "-":
#                 # pred/arg1 combination
#                 if not arg1.endswith(".n"):
#                     print("arg should be a noun but isn't", arg1)
#                 arg1 = arg1[:-2]
#                 arg1 = arg1.replace("_", " ")
#                 predarg1_vec[ (predicate, arg1) ] = np.array([ float(v) for v in pieces[1:]])

#             elif arg1 == "-":
#                 # pred/arg0 combination
#                 if not arg0.endswith(".n"):
#                     print("arg should be a noun but isn't", arg0)
#                 arg0 = arg0[:-2]
#                 arg0 = arg0.replace("_", " ")
#                 predarg0_vec[ (predicate, arg0) ] = np.array([ float(v) for v in pieces[1:]])

#             else:
#                 print("could not parse pred/arg entry, did not find - entry", label)
                

#         else:
#             # attribute
#             label = label.replace("|", " ")
#             attrib_vec[ label ] = np.array([ float(v) for v in pieces[1:]])


# for k in object_vec.keys(): print(k)

# from gensim.models import KeyedVectors
# kv = KeyedVectors(len(list(object_vec.values())[0]))
# kv.add_vectors(list(object_vec.keys()), list(object_vec.values()))
# kv.fill_norms()

# index_key = dict( (kv.get_index(key), key) for key in object_vec.keys())

# centroid = kv.get_mean_vector(["man", "woman"])
# # print(centroid)
# sims = kv.cosine_similarities(centroid, kv.get_normed_vectors())

# obj_sim = [ (index_key[ix], sim) for ix, sim in enumerate(sims)]
# obj_sim.sort(reverse = True, key = lambda p: p[1])
# for obj, sim in obj_sim[:20]:
#     print(obj, sim)
# print("----------")

# k0 = list(object_vec.keys())[0]
# print(k0, kv.similar_by_key(k0))


