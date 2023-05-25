## Katrin Erk May 2023
# utility for making symthetically ambiguous sentences
# based on vector similarity
#

import sys
import os
import zipfile
from collections import defaultdict, Counter
import math
import numpy as np
import random

import vgiterator
from sds_input_util import VGSentences, VGParam
from vgindex import VgitemIndex
from vec_util import VectorInterface

from vgpaths import VGPaths

class SyntheticPolysemes:
    def __init__(self, vgpath_obj, vgindex_obj, frequent_words, concept_scenario_mapping, max_fraction = 0.5):

        # store global data
        self.vec_obj = VectorInterface(vgpath_obj)
        self.vgix_obj = vgindex_obj
        self.vgfrequent_words = frequent_words
        self.concept_scenario_mapping = concept_scenario_mapping

        self.max_fraction = max_fraction

        # in rankings of similarity among words,
        # make bins of degrees of similarity
        self.simlevels = { 0 : [0, 14], 1:[15, 29], 2:[30, 44], 3:[45, 1000000]}

        # count how often each argument appears in the pred/arg entries in the vectors
        # so that we can later discard pred/arg pairs where the argument only ever
        # appears once (because then we'll never find a pair pred'/arg with the same arg
        # but different pred')
        self.arg0counter = Counter([a for _, a in self.vec_obj.predarg0_vec.keys()])
        self.arg1counter = Counter([a for _, a in self.vec_obj.predarg1_vec.keys()])


    ###
    # main method:
    # given a list of sentences,
    # make pseudo-polysemous entries for them
    # at the given similarity level.
    # words will be given IDs starting with next_wordid.
    #
    # returns: transformed sentences, along with
    # characterizations of the new polysemous "words"
    def make(self, sentences, next_wordid, simlevel = 3, singleword = False):

        gold = {}


        sentences_transformed = [ ]
        
        ctype_map = {"object" : "obj", "attr" : "att", "predarg0" : "rel", "predarg1" : "rel", "rel" : "rel"}


        for sentid, words, roles in sentences:


            ##
            # determine words that could be made into cloze items.
            # for roles, use curried words
            candidates = self._candidate_words(words, roles)

            if len(candidates) < 1:
                 sentences_transformed.append( [sentid, words, [], roles] )
                 continue
            ##
            # select candidates to make cloze items: if args.singleword, then one, 
            # else up to 1/2 of all words in the sentence
            numwords_for_cloze = 1 if singleword else random.randint(1, max(1, int(len(candidates) * self.max_fraction)))
            words_for_cloze = random.sample(candidates, numwords_for_cloze)

            ##
            # make cloze words, store in gold, store transformed literal

            targetwords = [ ]
            targetword_ids = [ ]

            for ctype, cdata, ix in words_for_cloze:
                # find second word as cloze partner for the word in cdata.
                # simlevel says whether to restrict similarities to a particular quartile in the ranked list
                # of vector neighbors for word1
                retv = self._sample_clozepair(ctype, cdata, simlevel)
                if retv is None:
                    # print("Unable to sample cloze pair for", cdata, ctype, ix)
                    continue
                word1, word1id, word2, word2id, rel_rank_of_word2 = retv

                # types recorded here: obj, att, rel

                gold[next_wordid] = {
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
            sentences_transformed.append( [sentid, otherwords, targetwords, roles] )

        return (sentences_transformed, gold)

    ##
    # making a cloze pair for one object
    def make_obj_cloze(self, conceptid, simlevel):
        conceptlabel, _ = self.vgix_obj.ix2l(conceptid)
        # can we make a cloze partner for this word?
        if conceptlabel not in self.vgfrequent_words["objects"] or conceptlabel not in self.vec_obj.object_vec.keys() or conceptid not in self.concept_scenario_mapping:
            return None

        retv = self._sample_clozepair("object", conceptlabel, simlevel)
        if retv is None:
            return None

        word1, word1id, word2, word2id, rel_rank_of_word2 = retv

        return (word1, word1id, word2, word2id)
            

    ##
    # make list of candidates for cloze.
    # objects, attributes: use as is.
    # relations: make two curried variants, with each argument
    def _candidate_words(self, words, roles):
        ###
        # store mappings dref-> object labels
        dref_obj = defaultdict(list)
        for _, conceptid, dref in words:
            conceptlabel, ctype = self.vgix_obj.ix2l(conceptid)
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
            conceptlabel, ctype = self.vgix_obj.ix2l(conceptid)

            if ctype == "obj":
                # object: no currying necessary. but is this frequent enough?
                # if conceptlabel not in self.vgfrequent_words["objects"]:
                #     print("rare object", conceptlabel)
                # if conceptlabel not in self.vec_obj.object_vec.keys():
                #     print("no vector for", conceptlabel)
                # if conceptid not in self.concept_scenario_mapping:
                #     print("no scenario for", conceptlabel, conceptid)
                if conceptlabel in  self.vgfrequent_words["objects"] and conceptlabel in self.vec_obj.object_vec.keys() and conceptid in self.concept_scenario_mapping:
                    # yes, usable
                    retv.append( ("object",conceptlabel, wordix) )

            elif ctype == "att":
                # attribute: no currying necessary. but is this frequent enough?
                if conceptlabel in  self.vgfrequent_words["attributes"] and conceptlabel in self.vec_obj.attrib_vec.keys() and conceptid in self.concept_scenario_mapping:
                    # yes, usable
                    retv.append( ( "attr", conceptlabel, wordix) )

            else:
                # relation: we need to curry this, if it's frequent enough
                if conceptlabel in self.vgfrequent_words["relations"] and conceptid in self.concept_scenario_mapping:
                    # may be usable if the argument is frequent enough
                    if dref not in arg0_drefh_drefd or dref not in arg1_drefh_drefd:
                        # no arguments recorded, skip
                        continue

                    # retain only objects that have a vector together with the predicate
                    # and that appear with at least 2 predicates in the vectors list
                    predarg0s = [ ("predarg0", conceptlabel, alabel) for alabel in dref_obj[ arg0_drefh_drefd[ dref ]] \
                                      if (conceptlabel, alabel) in  self.vec_obj.predarg0_vec.keys() and self.arg0counter[alabel] >= 10]
                    predarg1s = [ ("predarg1", conceptlabel, alabel) for alabel in dref_obj[ arg1_drefh_drefd[ dref]] \
                                      if (conceptlabel, alabel) in self.vec_obj.predarg1_vec.keys() and self.arg1counter[alabel] >= 10]


                    if len(predarg0s) > 0 or len(predarg1s) > 0:
                        retv.append( (ctype, predarg0s + predarg1s, wordix) )

        return retv


    #############3
    def _sample_clozepair(self, wordtype, wordinfo, simlevel):

        # determine ranked similarities
        if wordtype == "object":
            # object: wordinfo is simply a concept label

            word1 = wordinfo

            # compute similarity based on embeddings
            neighbors = [n for n, _ in self.vec_obj.ranked_sims(word1, wordtype)[1:] if self.vgix_obj.isobj(n)]

            # sample a 2nd word in simlevel, return its index and the percentile of the index
            word2index, word2rank = self._sample_cloze_fromlist(neighbors, simlevel)
            word2 = neighbors[word2index]

            # map word to word ID
            word1id = self.vgix_obj.o2ix(word1) 
            word2id = self.vgix_obj.o2ix(word2)

        elif wordtype == "attr":
            # object: wordinfo is simply a concept label

            word1 = wordinfo

            # compute similarity based on embeddings
            neighbors = [n for n, _ in self.vec_obj.ranked_sims(word1, wordtype)[1:] if self.vgix_obj.isatt(n)]

            # sample a 2nd word in simlevel, return its index and the percentile of the index
            word2index, word2rank = self._sample_cloze_fromlist(neighbors, simlevel)
            word2 = neighbors[word2index]

            # map word to word ID
            word1id = self.vgix_obj.a2ix(word1) 
            word2id = self.vgix_obj.a2ix(word2)

        elif wordtype == "rel":
            # relation: wordtype is a list of triples (predarg0/1, predlabel, arglabel)
            # sample one of the triples, determine neighbors
            argtype, pred1, arg1 = random.choice(wordinfo)
            word1 = pred1

            # filter neighbors: have to have different pred, same arg
            neighbors = [n[0] for n, _ in self.vec_obj.ranked_sims((pred1, arg1), argtype)[1:] if self.vgix_obj.isrel(n[0]) and n[0] != pred1 and n[1] == arg1]
            if len(neighbors) == 0:
                # print("No neighbors for pred/arg pair, skipping:", pred1, arg1, argtype)
                return None

            # sample a 2nd word in simlevel, return its index and the percentile of the index        
            word2index, word2rank = self._sample_cloze_fromlist(neighbors, simlevel)

            # what is the actual 2nd word? We had transformed neighbors to only retain pred before
            word2 = neighbors[word2index]

            # map word to word ID
            word1id = self.vgix_obj.r2ix(word1)
            word2id = self.vgix_obj.r2ix(word2)

        else:
            raise Exception( "unknown wordtype " + str(wordtype))


        if word1id is None:
            print("could not determine ID for", word1, wordtype)
            return None
        elif word2id is None:
            print("could not determine ID for", word2, wordtype)
            return None

        return (word1, word1id, word2, word2id, word2rank)


    def _sample_cloze_fromlist(self, neighbors, simlevel):
        if simlevel < 0:
            # simlevel not set?
            # choose one of the simlevels at random
            simlevel = random.choice(list(self.simlevels.keys()))

        simlevel_lower, simlevel_upper = self.simlevels[ simlevel]
        simlevel_upper = min([simlevel_upper, len(neighbors) -1])

        # select a neighbor, either from anywhere or from the selected level
        if len(neighbors) < 45:
            # sample from anywhere, we have too little data to distinguish levels
            index = random.randint(0, len(neighbors)-1)

        else:
            index = random.randint(simlevel_lower, simlevel_upper)


        word2rank = index / len(neighbors)

        return (index, word2rank)
