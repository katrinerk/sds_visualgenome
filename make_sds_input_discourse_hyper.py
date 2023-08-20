# Katrin Erk May 2023
# Write input for an SDS system:
# parameter files, and input passages (each consisting of multiple sentences)
#
# Here: multi-sentence discourse, with mental files,
# evaluation involving hypernymy
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
from hypernymy_util import HypernymHandler
from vec_util import VectorInterface


########3



parser = ArgumentParser()
parser.add_argument('--output', help="directory to write output to, default: sds_in/discourse", default = "sds_in/discourse/")
parser.add_argument('--vgdata', help="directory with VG data including frequent items, train/test split, topic model", default = "data/")
parser.add_argument('--orig', help="leave probe noun as is, rather than transform to hypernym? default: false", action = "store_true")
parser.add_argument('--test', help = "evaluate on test sentences rather than dev. Default: false.", action = "store_true")
parser.add_argument('--numpar', help="Number of paragraphs to use per chain type (4 chain types), default 300", type = int, default = 300)
parser.add_argument('--cond', help="Hypernymy condition. ch = concept/hypernymy weights. chh = plus hyper/hyper weights. chhi = plus incompatibility. default ch.", default = "ch")


args = parser.parse_args()

# get dev or test sentences?
if args.test:
    testsection = "test"
else:
    testsection = "dev"

# how many distractor sentences to use?
num_distractors = 4


# settings file
config = configparser.ConfigParser()
config.read("settings.txt")
selpref_method = config["Selpref"]


print("Testing on section:", testsection)
print("Selectional preferences:", selpref_method["Method"])
print("Transform probe noun to hypernym?", not(args.orig))
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

##
# vector space reader:
# only use objects that also have vectors.
vec_obj = VectorInterface(vgpath_obj)
available_objects = [o for o in vgobjects_attr_rel[VGOBJECTS] if o in vec_obj.object_vec]

#
# hypernymy object
first_hyperid = len(vgobjects_attr_rel[VGOBJECTS]) + len(vgobjects_attr_rel[VGATTRIBUTES]) + len(vgobjects_attr_rel[VGRELATIONS])
hyper_obj = HypernymHandler(available_objects, index_of_first_hypernym = first_hyperid) 


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
#
# in addition, determine all hypernyms, and hypernym IDs, of the core object concept ID, and
# also map chains of type 1,2,3,4 with the hypernym IDs instead of the object concept ID.
# we record separate chains for the actual object concepts
# and for their hypernyms
chain = {1 :defaultdict(list), 2: defaultdict(list), 3:defaultdict(list), 4:defaultdict(list) }
hchain = {1 :defaultdict(list), 2: defaultdict(list), 3:defaultdict(list), 4:defaultdict(list) }

for sentid, words, roles, in vgsent_obj.each_sentence(vgiter, vgobjects_attr_rel, traintest_split, testsection):
    # store the sentence
    sentid_sentence[ sentid ] = (words, roles)

    # get a representation more useful for indexing
    dref_names, dref_wordtype, pred_args, arg_preds = recode_sentence(words, roles, vgindex_obj)


    # for every object-denoting dref, check for chains
    for dref, wordtype in dref_wordtype.items():
        if wordtype != VGOBJECTS: continue

        # find all attributes and all relations of which it is the arg0
        thisword_attrib = [dref_h for role, dref_h in arg_preds[dref] if role == "arg1" and dref_wordtype[dref_h] == VGATTRIBUTES]
        thisword_rel = [dref_h for role, dref_h in arg_preds[dref] if role == "arg0"]

        # go through all labels of this discourse referent
        for word_id in dref_names[ dref]:
            # determine hypernyms
            word_label = vgindex_obj.ix2l(word_id)[0]

            # only index this word ID if it has hypernyms
            if word_label not in hyper_obj.object_hypernyms: continue
                
            hypernym_labels = hyper_obj.object_hypernyms[word_label]
            hypernym_ids = [hyper_obj.hypernym_index(ell) for ell in hypernym_labels]
            
            # definitely record a chain1 for each word label of this discourse referent
            chain[1][ (word_id,) ].append( (sentid, (dref,)) )
            # and also for hypernyms
            for h_id in hypernym_ids:
                hchain[1][ (h_id,)].append( (sentid, (dref,)) )

            # store chain2's starting with each attribute
            for dref_h in thisword_attrib:
                # only a single attribute label per attribute
                attrword_id = dref_names[dref_h][0]
                chain[2][(word_id, attrword_id)].append( (sentid, (dref, dref_h)) )
                # and with hypernyms of the central object label
                for h_id in hypernym_ids:
                    hchain[2][(h_id, attrword_id)].append( (sentid, (dref, dref_h)) )

            # store chain3's starting with each relation
            for dref_h in thisword_rel:
                # only a single arg1 per relation
                dref_d1 = [d for role, d in pred_args[dref_h] if role == "arg1"][0]
                # and only a single name per relation
                relword_id = dref_names[dref_h][0]
            
                for arg1word_id in dref_names[dref_d1]:
                
                    # store a chain3 for each name of the arg1 
                    chain[3][ (word_id, relword_id, arg1word_id ) ].append( (sentid, (dref, dref_h, dref_d1)) )
                    # and for hypernyms
                    for h_id in hypernym_ids:
                        hchain[3][ (h_id, relword_id, arg1word_id ) ].append( (sentid, (dref, dref_h, dref_d1)) )
                    
                    # and store a chain4 for each attribute
                    for attrib_dref_h in thisword_attrib:
                        # only a single attribute label per attribute
                        attrword_id = dref_names[attrib_dref_h][0]
                        chain[4][(word_id, attrword_id, relword_id, arg1word_id)].append( (sentid, (dref, attrib_dref_h, dref_h, dref_d1)) )
                        # ... and hypernyms
                        for h_id in hypernym_ids:
                            hchain[4][(h_id, attrword_id, relword_id, arg1word_id)].append( (sentid, (dref, attrib_dref_h, dref_h, dref_d1)) )
                    


##########################
# chain type specific functions

# given sampled labels, turn them into a definite description
def chain_labels_to_defdescr(labels, hyper_id, chaintype):
    if chaintype == 1:
        # cat -> cat(0)
        object_id = labels[0]
        return ([[["prew", object_id, 0]],   []],
                [[["prew", hyper_id, 0]],   []] )

    elif chaintype == 2:
        # black cat => cat(0) & black(1) & arg1(1,0)
        object_id, attrib_id = labels
        return ( [[ ["prew", object_id, 0], ["prew", attrib_id, 1]], [[ "prer", "arg1", 1, 0]]],
                 [[ ["prew", hyper_id, 0], ["prew", attrib_id, 1]], [[ "prer", "arg1", 1, 0]]] )

    elif chaintype == 3:
        # cat on roof => cat(0) & on(1) & roof(2) & arg0(1,0) & arg1(1, 2)
        obj0_id, rel_id, obj1_id = labels
        return ( [[ ["prew", obj0_id, 0], ["prew", rel_id, 1], ["prew", obj1_id, 2] ], [ ["prer", "arg0", 1, 0], ["prer", "arg1", 1, 2]] ],
                 [[ ["prew", hyper_id, 0], ["prew", rel_id, 1], ["prew", obj1_id, 2] ], [ ["prer", "arg0", 1, 0], ["prer", "arg1", 1, 2]] ]) 

    elif chaintype == 4:
        # black cat on roof => cat(0) & black(1) & on(2) & roof(3) & arg1(1,0) & arg0(2,0) & arg1(2,3)
        obj0_id, attrib_id, rel_id, obj1_id = labels
        return ( [[ ["prew", obj0_id, 0], ["prew", attrib_id, 1], ["prew", rel_id, 2], ["prew", obj1_id, 3] ],
                    [ ["prer", "arg1", 1, 0], ["prer", "arg0", 2, 0], ["prer", "arg1", 2, 3]] ],
                 [[ ["prew", hyper_id, 0], ["prew", attrib_id, 1], ["prew", rel_id, 2], ["prew", obj1_id, 3] ],
                    [ ["prer", "arg1", 1, 0], ["prer", "arg0", 2, 0], ["prer", "arg1", 2, 3]] ] )
        

# given sampled labels, return sentence IDs of possible distractor sentences
# returns:
# - list of (sentence IDs, drefs) of tricky distractors, which share some material with the target
# - list of sentence IDs of all distractors
def chain_distractor_candidates(labels, chain, sentid_sentence, chaintype, minlength = 2):
    # pull out all sentences that contain these exact labels,
    # as indexed in the correct chain 
    sentences_containing_target = set(sentid for sentid, dref in chain[chaintype][labels])
    # pull out all sentences that do not contain these exact labels
    # if the target labels contain an object hypernym, this should be all sentences
    # that do not contain any sequence in which an object concept could be replaced by its hypernym
    # to produce the target labels
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
# create a mapping:
# from pairs (sentence ID, discourse referent) in the sentence with
#   the definite description
# to pairs (sentence ID, discourse referent) in the sentence
#   to which the definite description refers.
# In the definite description sentence, the discourse referents are 0, 1, 2, ...
# because there's nothing else in the sentence.
# In the target sentence, the discourse referents are whatever is given,
# as the target wsentence is from the VG originally
# 
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
                 
##
# for a given sampled expression,
# identify the central object concept,
# randomly sample a hypernym,
# and return its ID
def choose_hypernym(labels, vgindex_obj, hyper_obj):
    conceptid = labels[0]
    conceptlabel, _ = vgindex_obj.ix2l(conceptid)

    if conceptlabel not in hyper_obj.object_hypernyms:
        # we don't have any hypernyms for this object.
        # this shouldn't have happened, but here we are
        raise Exception("shouldn't be here " + conceptlabel)
    
    # randomly choose a hypernym candidate
    hypercandidates = hyper_obj.object_hypernyms[conceptlabel]
    hyperlabel = random.choice(hypercandidates)
    hyper_id = hyper_obj.hypernym_index(hyperlabel)

    new_labels = tuple([hyper_id] + list(labels[1:]))

    return (conceptid, hyper_id, new_labels)
      
######################3
# sample passages for evaluation:
# - sample a definite description to target
# - sample a target sentence containing a matching indefinite description
# - sample distractor sentences
print("Sampling passages")

# gold info:
# "discourse" ->
#   "cond" -> "hyper",
#   "subcond" -> "hyper" or "orig", for hypernym or original word
#   "passages" -> list of dictionaries
gold = { "discourse" : { "cond" : "hyper",
                         "subcond" :  [ "hyper", "orig"][int(args.orig)],
                         "passages" : [ ] }} 


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
        # this is from the original phrases, without hypernyms.
        sampled_labels = random.choice(list(chain[chainlength].keys()))

        # identify a hypernym ID, and
        # adapted labels exchanging the central object concept by its hypernym
        objconcept_id, hyper_id, hyper_labels = choose_hypernym(sampled_labels, vgindex_obj, hyper_obj)
        
        # turn it into a definite description:
        # words, roles
        # HIER should we take the hypernym into account?
        definite_descr, hyper_definite_descr = chain_labels_to_defdescr(sampled_labels, hyper_id, chainlength)

        # sample a sentence containing a matching indefinite description:
        # sample based on the original object label, not the hypernym,
        # such that we can be sure that the target sentence contains the exact
        # same object label, not another object label with the same hypernym
        target_sentid, target_drefs = random.choice(list(chain[chainlength][ sampled_labels]))
        objconcept_dref = target_drefs[0]

        # sample k distractor sentences, half of them tricky.
        # but sample using the hypernym
        # to make sure the distractors don't match it 
        tricky_candidates, other_candidates  = chain_distractor_candidates(hyper_labels, hchain, sentid_sentence, chainlength)
        ## TESTING: do we have enough tricky candidates to make interesting distractors?
        # yes, looks like we do.
        # print("chain length", chainlength, "num tricky candidates", len(tricky_candidates))
        
        num_tricky_distractors = min(int(num_distractors/2), len(tricky_candidates))
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
                                               "hyper_descr" : hyper_definite_descr,
                                               "hyper" : {"objconcept_id" : objconcept_id,
                                                          "hyper_id" : hyper_id,
                                                          "dref" : objconcept_dref,
                                                          "sentid" : target_sentid},
                                               "chain" : chainlength} )

        # TESTING:
        # do the target sentences all include the target expression?
        # do none of the distractor sentences include a word that
        # has the the exact same hypernym as the central object concept,
        # in exactly the phrase of the target expression?
        # print("----------")
        # def conclabel_w_hypermark(oid, hyper_id, vgindex_obj, hyper_obj):
        #     conceptlabel, ctype = vgindex_obj.ix2l(oid)
        #     if ctype != VGOBJECTS:
        #         return conceptlabel
        #     if conceptlabel not in hyper_obj.object_hypernyms:
        #         return conceptlabel

        #     hyperlabel = hyper_obj.hypernym_ixlabel(hyper_id)
        #     if hyperlabel in hyper_obj.object_hypernyms[conceptlabel]:
        #         return "**" + conceptlabel + "**"
        #     else:
        #         return conceptlabel
            
        # print("obj concept", vgindex_obj.ix2l(objconcept_id)[0], "hyper concept", hyper_obj.hypernym_ixlabel(hyper_id), "dref", objconcept_dref, "sent", target_sentid)
        # print("definite descr", [vgindex_obj.ix2l(ell)[0] for ell in sampled_labels])
        # print()
        # print("TGT", [(vgindex_obj.ix2l(ell[1]), ell[2]) for ell in sentid_sentence[target_sentid][0]])
        # print()
        # for sentid, _ in distractors:
        #     print("DIS", [(conclabel_w_hypermark(ell[1], hyper_id, vgindex_obj, hyper_obj), ell[2]) for ell in sentid_sentence[sentid][0]])
        

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
                # remove the other ones.
                # also remove occurrences of the selected hypernym

                literals_to_remove = [ ]
                for other_sentid, other_drefs in hchain[chainlength][hyper_labels]:
                    if other_sentid == target_sentid and other_drefs != target_drefs:
                        # a non-target occurrence of the target expression (with the core object concept
                        # possibly replaced by the sampled hypernym) has been found
                        # in the target sentence.
                        # remove the literals in that non-target occurrence
                        
                        literals_to_remove = [ ]
                        
                        # combine the sampled labels with the other set of discourse referents
                        labels_and_drefs = list(zip(sampled_labels, other_drefs))
                        
                        # for the core object concept, remove any literals associated with the discourse referent in question
                        # that have the right hypernym, but that aren't the actual core object concept
                        mainconc_id, mainconc_dref = labels_and_drefs[0]
                        hyper_name = hyper_obj.hypernym_ixlabel(hyper_id)
                        for _, this_id, this_dref in words:
                            if this_dref == mainconc_dref and this_id != mainconc_id:
                                this_name, this_ctype = vgindex_obj.ix2l(this_id)
                                if this_ctype == VGOBJECTS and hyper_name in hyper_obj.object_hypernyms.get(this_name, []):
                                    # this is an object label associated with the discourse referent of the
                                    # main concept in this expression, and
                                    # it has the sampled hypernym as a hypernym: remove it
                                    literals_to_remove.append( [ "w", this_id, this_dref ] )
                        
                        
                        # and straight up remove the literals other than those of the core object concept
                        for ell, d in labels_and_drefs[1:]:
                            # only remove the literal if it is not also part of the target expression
                            if d not in target_drefs:
                                literals_to_remove.append( [ "w", ell, d ])
                        
                if len(literals_to_remove) > 0:
                    # # TESTING: which literals are we removing?
                    # print("------")
                    # print("obj concept", vgindex_obj.ix2l(objconcept_id)[0], "hyper concept",
                    #       hyper_obj.hypernym_ixlabel(hyper_id), "drefs", target_drefs, "sent", target_sentid)
                    # print("definite descr", [vgindex_obj.ix2l(ell)[0] for ell in sampled_labels])
                    # print("removing", [(vgindex_obj.ix2l(ell)[0], d) for _, ell, d in literals_to_remove])

                    
                    words, roles = vgsent_obj.remove_from_sent(words, roles, literals = literals_to_remove, keep_these_words= keep_words)
                    # re-separate into words to keep, and words to remove
                    keep_words = [[w, wordid, dref] for w, wordid, dref in words if dref in drefs_keep]
                    otherwords = [w for w in words if w not in keep_words]
                    
                    
            thispassage.append([sentid, otherwords, keep_words, roles])

        # add a sentence that just has the definite description to be matched
        # store the words of the definite description as "words to keep"
        usethis_descr = definite_descr if args.orig else hyper_definite_descr
        ddescr_words, ddescr_roles = usethis_descr

        # TESTING: no definite descriptions
        # ddescr_words = [ ["w", ell, d] for _, ell, d in ddescr_words]
        # ddescr_roles = [ ["r", a, d1, d2] for _, a, d1, d2 in ddescr_roles]
        
        thispassage.append( [ ddescr_sentid, [], ddescr_words, ddescr_roles] )

        # test printout
        # print("definite description", [ (vgindex_obj.ix2l(ell)[0] if ell < vgindex_obj.lastix else hyper_obj.hypernym_ixlabel(ell), d) for _, ell, d in ddescr_words])

        # and store the passage we've assembled, to be written out later
        passages.append(thispassage)

############3
print("computing general parameters")
vgparam_obj = VGParam(vgpath_obj, selpref_method, frequentobj = vgobjects_attr_rel)
global_param, scenario_concept_param, word_concept_param, selpref_param = vgparam_obj.get()

# adapt general parameters to the presence of hypernyms
global_param, scenario_concept_param, word_concept_param, selpref_param = hyper_obj.adapt_sds_params(global_param, scenario_concept_param,
                                                                                                     word_concept_param, selpref_param, vgindex_obj)
# compute additional hypernymy parameters
hyper_param = hyper_obj.compute_hyper_param(global_param, vgindex_obj, vec_obj, condition = args.cond)

####################
# write data
print("writing data")

# write general parameters
vgparam_obj.write(global_param, scenario_concept_param, word_concept_param, selpref_param)

# HIER where to write hypernymy parameters?

# write passages
print("writing", len(passages), "passages")
vgsent_obj.write_discourse(passages)

# write gold info
gold_zipfile, gold_file = vgpath_obj.sds_gold( write = True)
with zipfile.ZipFile(gold_zipfile, "w", zipfile.ZIP_DEFLATED) as azip:
    azip.writestr(gold_file, json.dumps(gold))


