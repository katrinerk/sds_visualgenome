# Katrin Erk May 2023
# Write input for an SDS system:
# parameter files, and input passages (each consisting of multiple sentences)
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
# import sentence_util



from vgnames import VGOBJECTS, VGATTRIBUTES, VGRELATIONS 
import vgiterator
from sds_input_util import VGSentences, VGParam
from vgindex import VgitemIndex
from vgpaths import VGPaths
from polysemy_util import SyntheticPolysemes
from hypernymy_util import HypernymHandler

########3



parser = ArgumentParser()
parser.add_argument('--output', help="directory to write output to, default: sds_in/discourse", default = "sds_in/discourse/")
parser.add_argument('--vgdata', help="directory with VG data including frequent items, train/test split, topic model", default = "data/")
parser.add_argument('--cond', help="condition: vanilla, poly, tpoly (target polysemy only), default: vanilla", default = "vanilla")
parser.add_argument('--test', help = "evaluate on test sentences rather than dev. Default: false.", action = "store_true")
parser.add_argument('--numpar', help="Number of paragraphs to use per chain type (4 chain types), default 300", type = int, default = 300)
parser.add_argument('--simlevel', help="choose cloze pairs only in this similarity range: 0-3, 0 is most similar, default: 1", type =int, default = 1)


args = parser.parse_args()

# get dev or test sentences?
if args.test:
    testsection = "test"
else:
    testsection = "dev"

if args.cond not in ["vanilla", "poly", "tpoly"]:
    print("Unknown condition", args.cond, "needs to be one of vanilla, poly, tpoly")
    sys.exit(1)

# how many distractor sentences to use?
num_distractors = 4


# settings file
config = configparser.ConfigParser()
config.read("settings.txt")
selpref_method = config["Selpref"]


print("Testing on section:", testsection)
print("Selectional preferences:", selpref_method["Method"])
print("Condition:", args.cond)
print("Num passages per chain type:", args.numpar)

print("reading data")

vgpath_obj = VGPaths(vgdata = args.vgdata,  sdsdata = args.output)

# frequent obj/attr/rel
vgcounts_zipfilename, vgcounts_filename = vgpath_obj.vg_counts_zip_and_filename()
with zipfile.ZipFile(vgcounts_zipfilename) as azip:
    with azip.open(vgcounts_filename) as f:
        vgobjects_attr_rel = json.load(f)
        
vgindex_obj = VgitemIndex(vgobjects_attr_rel)
vgiter = vgiterator.VGIterator()

# training/test split
split_zipfilename, split_filename = vgpath_obj.vg_traintest_zip_and_filename()
with zipfile.ZipFile(split_zipfilename) as azip:
    with azip.open(split_filename) as f:
        traintest_split = json.load(f)
    
########
# read test sentences, index them by 
# length-1, length-2, length-3 chains of entities
# connected by relations 
print("indexing sentences")

vgsent_obj = VGSentences(vgpath_obj)

##
# for a given sentence, make mapping from discourse referent to names,
# make a list of all the object-denoting discourse referents,
# and make a mapping from predicate drefs to argument drefs
# (and vice versa)
#
# returns:
# dref_names: mapping dref(number) -> list of unary labels that go with it (numbers)
# dref_wordtype: mapping dref(number) -> VGOBJECTS, VGATTRIBUTES, VGRELATIONS
# pred_args: mapping head dref -> list of pairs (role, dependent dref)
# arg_preds: mapping dependent dref -> list of pairs (role, head dref)
def recode_sentence(words, roles, vgindex_obj):
    dref_names = defaultdict(list) # keep names for discourse referent
    dref_wordtype = { } # obj or rel or attr

    for _, word_id, dref in words:
        word, wordtype = vgindex_obj.ix2l(word_id)
        if wordtype is None:
            raise Exception('failed to look up word ID' + str(word_id))

        dref_names[dref].append( word_id)
        dref_wordtype[dref] = wordtype

    pred_args = defaultdict(list)
    arg_preds = defaultdict(list)
    for _, role, dref_h, dref_d in roles:
        pred_args[dref_h].append( (role, dref_d))
        arg_preds[dref_d].append( (role, dref_h))

    return (dref_names, dref_wordtype, pred_args, arg_preds)

##############
# mapping sentence ID -> pair (words, roles)
sentid_sentence = { }
# chains:
# * chain of the form "cat":
#    mapping from single concept ID to list of pairs (sentence ID, discourse referent ID)
# * chain of the form "black cat":
#    mapping from a a pair (object ID, attribute ID)
#    to a list of tuples (sentence ID, object discourse ref ID, attribute discourse ref ID)
# * chain of the form "cat on roof":
#    mapping from tuple (object ID, relation ID, object ID)
#    to a list of tuples (sentence ID, arg0 object dref, relation dref, arg1 object dref)
# * chain of the form "black cat on roof":
#    mapping from tuple (object ID, attribute ID, relation ID, object ID)
#    to a list of tuples (sentence ID, arg0 object dref, attribute dref, relation dref, arg1 object dref)
chain = {1 :defaultdict(list), 2: defaultdict(list), 3:defaultdict(list), 4:defaultdict(list) }

for sentid, words, roles, in vgsent_obj.each_sentence(vgiter, vgobjects_attr_rel, traintest_split, testsection):
    # store the sentence
    sentid_sentence[ sentid ] = (words, roles)

    # get a representation more useful for indexing
    dref_names, dref_wordtype, pred_args, arg_preds = recode_sentence(words, roles, vgindex_obj)

    # for every object-denoting dref, check for chains
    for dref, wordtype in dref_wordtype.items():
        if wordtype != VGOBJECTS: continue

        # definitely record a chain1 for each word label of this discourse referent
        for word_id in dref_names[ dref]:
            chain[1][ (word_id,) ].append( (sentid, (dref,)) )

        # find all attributes and all relations of which it is the arg0
        thisword_attrib = [dref_h for role, dref_h in arg_preds[dref] if role == "arg1" and dref_wordtype[dref_h] == VGATTRIBUTES]
        thisword_rel = [dref_h for role, dref_h in arg_preds[dref] if role == "arg0"]

        # store chain2's starting with each attribute
        for dref_h in thisword_attrib:
            # only a single attribute label per attribute
            attrword_id = dref_names[dref_h][0]
            chain[2][(word_id, attrword_id)].append( (sentid, (dref, dref_h)) )

        # store chain3's starting with each relation
        for dref_h in thisword_rel:
            # only a single arg1 per relation
            dref_d1 = [d for role, d in pred_args[dref_h] if role == "arg1"][0]
            # and only a single name per relation
            relword_id = dref_names[dref_h][0]
            
            for arg1word_id in dref_names[dref_d1]:
                
                # store a chain3 for each name of the arg1 
                chain[3][ (word_id, relword_id, arg1word_id ) ].append( (sentid, (dref, dref_h, dref_d1)) )

                # and store a chain4 for each attribute
                for attrib_dref_h in thisword_attrib:
                    # only a single attribute label per attribute
                    attrword_id = dref_names[attrib_dref_h][0]
                    chain[4][(word_id, attrword_id, relword_id, arg1word_id)].append( (sentid, (dref, attrib_dref_h, dref_h, dref_d1)) )
                    
        
##########################
# chain type specific functions

# given sampled labels, turn them into a definite description
def chain_labels_to_defdescr(labels, chaintype):
    if chaintype == 1:
        # cat -> cat(0)
        object_id = labels[0]
        return [[["prew", object_id, 0]],   []]

    elif chaintype == 2:
        # black cat => cat(0) & black(1) & arg1(1,0)
        object_id, attrib_id = labels
        return [[ ["prew", object_id, 0], ["prew", attrib_id, 1]], [[ "prer", "arg1", 1, 0]]]

    elif chaintype == 3:
        # cat on roof => cat(0) & on(1) & roof(2) & arg0(1,0) & arg1(1, 2)
        obj0_id, rel_id, obj1_id = labels
        return [[ ["prew", obj0_id, 0], ["prew", rel_id, 1], ["prew", obj1_id, 2] ], [ ["prer", "arg0", 1, 0], ["prer", "arg1", 1, 2]] ]

    elif chaintype == 4:
        # black cat on roof => cat(0) & black(1) & on(2) & roof(3) & arg1(1,0) & arg0(2,0) & arg1(2,3)
        obj0_id, attrib_id, rel_id, obj1_id = labels
        return [[ ["prew", obj0_id, 0], ["prew", attrib_id, 1], ["prew", rel_id, 2], ["prew", obj1_id, 3] ],
                [ ["prer", "arg1", 1, 0], ["prer", "arg0", 2, 0], ["prer", "arg1", 2, 3]] ]
        

# given sampled labels, return sentence IDs of possible distractor sentences
# returns:
# - list of (sentence IDs, drefs) of tricky distractors, which share some material with the target
# - list of sentence IDs of all distractors
def chain_distractor_candidates(labels, chain, sentid_sentence, chaintype, minlength = 2):
    sentences_containing_target = set(sentid for sentid, dref in chain[chaintype][labels])
    other_sent = [(s, []) for s in sentid_sentence.keys() if s not in sentences_containing_target]

    
    if chaintype == 1:
        # distractors: all sentences that don't contain the target object iD
        # there are no sentences that have particularly tricky distractors
        return (other_sent, other_sent)

    elif chaintype == 2:
        # "black cat".
        # tricky distractors: sentences with "cat".
        object_id, _ = labels
        tricky_distractors = aux_chain2distractors(object_id, chain, sentences_containing_target)

        return (tricky_distractors, other_sent)

    elif chaintype == 3:
        # "cat on roof"
        # tricky distractors: "cat on _".
        # if we cannot find enough, take distractors of the form "cat"
        obj0_id, rel_id,obj1_id  = labels
        tricky_distractors = aux_chain3distractors(obj0_id, rel_id,obj1_id, chain, sentences_containing_target)
        
        if len(tricky_distractors) < minlength:
            # we didn't get enough tricky distractors, add simpler ones
            tricky_distractors += aux_chain2distractors(obj0_id, chain, sentences_containing_target)

        return (tricky_distractors, other_sent)

    elif chaintype == 4:
        # black cat on roof
        # tricky distractors: "cat on roof"
        obj0_id, _, rel_id, obj1_id = labels
        tricky_distractors = [(s, drefs) for s, drefs in chain[3][(obj0_id, rel_id, obj1_id)] if s not in sentences_containing_target]

        if len(tricky_distractors) < minlength:
            # we didn't get enough tricky distractors, add simpler ones
            tricky_distractors += aux_chain3distractors(obj0_id, rel_id, obj1_id, chain, sentences_containing_target)

        if len(tricky_distractors) < minlength:
            tricky_distractors += aux_chain2distractors(obj0_id, chain, sentences_containing_target)
        
        # out_obj = sentence_util.SentencePrinter(vgindex_obj)
        # print("*****************\n", "---- chaintype 4, target", " ".join([vgindex_obj.ix2l(ell)[0] for ell in labels]), "\n")
        # for s in tricky_distractors[:5]:
        #     print("========\n", "sentid", s)
        #     out_obj.write_sentence(sentid_sentence[s][0] + sentid_sentence[s][1])
                
        return (tricky_distractors, other_sent)

##
# helpers
def aux_chain2distractors(object_id, chain, sentences_containing_target):
    return [(s, drefs) for s, drefs in chain[1][(object_id,)] if s not in sentences_containing_target]

def aux_chain3distractors(obj0_id, rel_id,obj1_id, chain, sentences_containing_target):
    poss_keys = [ell for ell in chain[3] if ell[0] == obj0_id and ell[1] == rel_id and ell[2] != obj1_id]
    return [(s, drefs) for ell in poss_keys for s, drefs in chain[3][ell] if s not in  sentences_containing_target]
        
########        
# from the target sentence ID and discourse referent list,
# create a mapping
# [[definite descr sentid, dref] , [target sentid, dref]  ]
def chain_dref_mappings(target_sentid, target_drefs, ddescr_sentid, chaintype):
    if chaintype == 1:
        # target_drefs is a single discourse referent
        target_dref = target_drefs[0]
        return [ [[ddescr_sentid, 0],  [target_sentid, target_dref]] ]

    elif chaintype == 2:
        # target_drefs is object_dref, attrib_dref
        obj_dref, attr_dref = target_drefs
        return [ [[ ddescr_sentid, 0], [target_sentid, obj_dref]],
                 [[ ddescr_sentid, 1], [target_sentid, attr_dref]] ]

    elif chaintype == 3:
        # target-drefs is object0_dref, rel_dref, object1_dref
        obj0_dref, rel_dref, obj1_dref = target_drefs
        return [ [[ ddescr_sentid, 0], [target_sentid, obj0_dref]],
                 [[ ddescr_sentid, 1], [target_sentid, rel_dref]],
                 [[ ddescr_sentid, 2], [target_sentid, obj1_dref]] ]

    elif chaintype == 4:
        # target-drefs is obj0_dref attrib_dref rel_dref obj1_dref
        obj0_dref, attr_dref, rel_dref, obj1_dref = target_drefs
        return [ [[ ddescr_sentid, 0], [target_sentid, obj0_dref]],
                 [[ ddescr_sentid, 1], [target_sentid, attr_dref]],
                 [[ ddescr_sentid, 2], [target_sentid, rel_dref]],
                 [[ ddescr_sentid, 3], [target_sentid, obj1_dref]] ]
                 

        
######################3
# sample passages for evaluation:
# - sample a definite description to target
# - sample a target sentence containing a matching indefinite description
# - sample distractor sentences
print("Sampling passages")

# gold info:
# "discourse" ->
#   "cond" -> condition
#   "passages" -> list of dictionaries
gold = { "discourse" : { "cond" : args.cond, "passages" : [ ] }}

# store the passages for writing
passages = [ ]

# sentence IDs for sentences to be added with
# definite descriptions:
# use sentence IDs not used before
next_probe_sentid = max([int(sid) for sidlist in traintest_split.values() for sid in sidlist])+ 1

# number of paragraphs times: sample a paragraph
random.seed(92857)

for chainlength in [1,2,3,4]:
    for _ in range(args.numpar):
        # sample a definite description to target of the right length
        sampled_labels = random.choice(list(chain[chainlength].keys()))
        # turn it into a definite description:
        # words, roles
        definite_descr = chain_labels_to_defdescr(sampled_labels, chainlength)

        # sample a sentence containing a matching indefinite description
        target_sentid, target_drefs = random.choice(list(chain[chainlength][ sampled_labels]))

        # sample k distractor sentences, half of them tricky
        tricky_candidates, other_candidates  = chain_distractor_candidates( sampled_labels, chain, sentid_sentence, chainlength)
        num_tricky_distractors = min(int(num_distractors/2), len(tricky_candidates))
        # print("num candidate tricky distractors", len(tricky_candidates), "chainlength", chainlength)
        distractors = random.sample(tricky_candidates, k = num_tricky_distractors) if num_tricky_distractors > 0 else [ ]
        distractors += random.sample( [s for s in other_candidates if s not in distractors], k = num_distractors - num_tricky_distractors)

        ddescr_sentid = str(next_probe_sentid)
        next_probe_sentid += 1

        candidate_sents = [(target_sentid, target_drefs)]+ distractors
        random.shuffle(candidate_sents)

        # store this choice in gold
        gold["discourse"]["passages"].append( {"sentids": [sentid for sentid, _ in candidate_sents]  + [ddescr_sentid],
                                               "target" : chain_dref_mappings(target_sentid, target_drefs, ddescr_sentid, chainlength),
                                               "descr" : definite_descr,
                                               "chain" : chainlength} )

        # print("HIER0 sentence IDs:", [sentid for sentid, _ in candidate_sents]  + [ddescr_sentid])

        # if chainlength == 4:
        #     ddescr_words, ddescr_roles = definite_descr
        #     print("HIER1 definite descr", [(vgindex_obj.ix2l(w)[0], d) for _, w, d in ddescr_words])
        #     print("HIER2 target sent full", [(vgindex_obj.ix2l(w)[0], d) for _, w, d in sentid_sentence[target_sentid][0]])

        # construct the passage
        thispassage = [ ]
        for sentid, drefs_keep in candidate_sents:
            words, roles = sentid_sentence[sentid]

            # make sure to keep the target words, and all distractor words
            # chosen to make the task tricky
            keep_words = [[w, wordid, dref] for w, wordid, dref in words if dref in drefs_keep]
            otherwords = [w for w in words if w not in keep_words]
            
            if sentid == target_sentid:
                # if the sentence contains multiple occurrences of the target word,
                # remove the other ones

                literals_to_remove = [ ]
                for other_sentid, other_drefs in chain[chainlength][sampled_labels]:
                    if other_sentid == target_sentid and other_drefs != target_drefs:
                        remove_labels_drefs = [(ell, d) for ell, d in zip(sampled_labels, other_drefs) if d not in target_drefs]
                        
                        for ell, d in remove_labels_drefs:
                            literals_to_remove.append( ["w", ell, d] )

                if len(literals_to_remove) > 0:
                    words, roles = vgsent_obj.remove_from_sent(words, roles, literals = literals_to_remove, keep_these_words= keep_words)
                    # re-separate into words to keep, and words to remove
                    keep_words = [[w, wordid, dref] for w, wordid, dref in words if dref in drefs_keep]
                    otherwords = [w for w in words if w not in keep_words]
                    
                    
            thispassage.append([sentid, otherwords, keep_words, roles])

        # add a sentence that just has the definite description to be matched
        # store the words of the definite description as "words to keep"
        ddescr_words, ddescr_roles = definite_descr
        thispassage.append( [ ddescr_sentid, [], ddescr_words, ddescr_roles] )

        # and store the passage we've assembled, to be written out later
        passages.append(thispassage)

# sys.exit(1)
############3
print("computing general parameters")
vgparam_obj = VGParam(vgpath_obj, selpref_method, frequentobj = vgobjects_attr_rel)
global_param, scenario_concept_param, word_concept_param, selpref_param = vgparam_obj.get()

#################
# transform passages according to evaluation condition:
# - vanilla: nothing
# - poly: add polysemy to about half the unary literals
# - tpoly: add polysemy to each word of the probe. add another
#     sentence at the end with the probe as an indefinite description, and
#     the same polysemy.
print("adapting data to the condition", args.cond)
if args.cond == "vanilla":
    # nothing to be done
    pass
elif args.cond == "poly":
    # add in polysemy at random in all sentences
    poly_obj = SyntheticPolysemes(vgpath_obj, vgindex_obj, vgobjects_attr_rel, scenario_concept_param)
    
    next_wordid = len(vgobjects_attr_rel[VGOBJECTS]) + len(vgobjects_attr_rel[VGATTRIBUTES]) + len(vgobjects_attr_rel[VGRELATIONS])
    poly_obj.initialize_stepwise(next_wordid, simlevel = args.simlevel)
    new_passages =  [ ]
    for passage in passages:
        newpassage = [ ]
        for sentid, words, targetwords, roles in passage:
            # separately transform words and targetwords. roles don't get transformed anyway.
            # old... are words that didn't get polysemized.
            # new... are words that did get synthetic polysemy.
            _, oldwords, newwords, _, = poly_obj.make_stepwise([ sentid, words, [ ]])
            _, oldtargetwords, newtargetwords, _= poly_obj.make_stepwise( [sentid, targetwords, [] ] )

            newpassage.append( [ sentid, oldwords, newtargetwords + oldtargetwords + newwords, roles] )

        new_passages.append(newpassage)

    # extract gold words
    goldwords = poly_obj.finalize_stepwise()
    passages = new_passages

    # record them in global parameters
    num_clozewords = len(goldwords)
    global_param["num_words"] += num_clozewords

    # add to word-concept log probabilities:
    # cloze word IDs -> concept ID -> logprob
    for word_id, entry in goldwords.items():
        # word cloze ID -> concept ID -> log of 0.5:
        # equal output probability for both concepts
        word_concept_param[ word_id ] = dict( (entry["concept_ids"][i],  -0.69) for i in [0,1])

    # and record them in the gold data
    gold["cloze"] = goldwords

    
elif args.cond == "tpoly":
    # add in polysemy in the sentence with the definite description,
    # and duplicate that sentence into an indefinite
    poly_obj = SyntheticPolysemes(vgpath_obj, vgindex_obj, vgobjects_attr_rel, scenario_concept_param, all_polysemous = True)
    
    next_wordid = len(vgobjects_attr_rel[VGOBJECTS]) + len(vgobjects_attr_rel[VGATTRIBUTES]) + len(vgobjects_attr_rel[VGRELATIONS])
    poly_obj.initialize_stepwise(next_wordid, simlevel = args.simlevel)
    new_passages =  [ ]
    for passage in passages:
        # everything except the probe sentence, keep unchanged
        newpassage = passage[:-1]
        # no non-keep words in that sentence
        sentid, _, targetwords, roles = passage[-1]

        # add polysemy
        _, oldtargetwords, newtargetwords, roles = poly_obj.make_stepwise([ sentid, targetwords, roles])
        newpassage.append( [ sentid, [], oldtargetwords + newtargetwords, roles] )

        # and duplicate the sentence, but make indefinite
        newpassage.append( [sentid, [],
                            [ ["w", wordid, dref] for _, wordid, dref in oldtargetwords + newtargetwords],
                            [ ["r", role, d1, d2] for _, role, d1, d2 in roles]] )

        new_passages.append(newpassage)


    # extract gold words
    goldwords = poly_obj.finalize_stepwise()
    passages = new_passages

    # record them in the global data
    num_clozewords = len(goldwords)
    global_param["num_words"] += num_clozewords

    # add to word-concept log probabilities:
    # cloze word IDs -> concept ID -> logprob
    for word_id, entry in goldwords.items():
        # word cloze ID -> concept ID -> log of 0.5:
        # equal output probability for both concepts
        word_concept_param[ word_id ] = dict( (entry["concept_ids"][i],  -0.69) for i in [0,1])
         

    # and store in gold
    gold["cloze"] = goldwords



else:
    raise Exception("shouldn't be here")


####################
# write data
print("writing data")

# write general parameters
vgparam_obj.write(global_param, scenario_concept_param, word_concept_param, selpref_param)
# write passages
print("writing", len(passages), "passages")
vgsent_obj.write_discourse(passages)

# write gold info
gold_zipfile, gold_file = vgpath_obj.sds_gold( write = True)
with zipfile.ZipFile(gold_zipfile, "w", zipfile.ZIP_DEFLATED) as azip:
    azip.writestr(gold_file, json.dumps(gold))

