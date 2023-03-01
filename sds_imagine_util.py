## Katrin Erk Feb 2023
# Utilities for imagining objects from scenarios,
# and attributes from concepts,
# to be re-used across
# evaluation scripts and interactive SDS

import os
import json
import zipfile
import math
from collections import Counter
import numpy as np

class ImagineScen:
    def __init__(self, vgpath_obj, vgindex_obj):

        # read scenario/topic data from gensim.
        # format: one row per topic, one column per word, log probabilities
        gensim_zipfilename, overall_filename, topic_filename, word_filename, topicword_filename = vgpath_obj.gensim_out_zip_and_filenames()
        with zipfile.ZipFile(gensim_zipfilename) as azip:
            # term-topic matrix
            with azip.open(topicword_filename) as f:
                term_topic_matrix = json.load(f)

            # ordered list of words as they appear in topics,
            # need to be mapped to IDs
            # ordered list of words as they appear in topics. need to be mapped to concepts and concept indices
            with azip.open(word_filename) as f:
                topic_wordlist = json.load(f)

        # make list of indices of gensim words that are objects,
        # and map object names ot their IDs
        self.indices_of_objects = [ ]
        self.object_ids = [ ]
        for index, word in enumerate(topic_wordlist):
            if word[:3] =="obj":
                wordid = vgindex_obj.o2ix(word[3:])
                if wordid is None:
                    raise Exception("lookup error for topic word, object", word[3:])

                self.object_ids.append(wordid)
                self.indices_of_objects.append(index)

        self.indices_of_objects = np.array(self.indices_of_objects)

        # for each scenario, restrict to objects, renormalize, store
        self.scenario_logprobs = [ ]
        for sclp in term_topic_matrix:

            a = np.array(sclp)

            # select only columns that describe an object
            a = a[ self.indices_of_objects ]

            # renormalize
            normalizer = np.log(np.exp(a).sum())
            a = a - normalizer
            # scenario_logprobs: scenario log probabilities, in order of scenarios
            self.scenario_logprobs.append(a)


    def predict_objectids(self, scenario_list):
        # obtain scenario probabilities within the scenario list
        sc_freq = Counter(scenario_list)
        sc_logprob = dict( (s, math.log( sc_freq[s] / sum(sc_freq.values()))) for s in sc_freq.keys() )

        # sc_logprob: dictionary scenario ID -> log probability of this scenario in this sentence
        # scenario_logprobs: list where the i-th entry is for the i-th scenario,
        #    each entry is a numpy array of object log probabilities
        # compute: sum_{s scenario in this sentence} this_sentence_scenarioprob(s) * object_probs(s)
        # again a numpy array of object log probabilities
        objectid_logprob = np.sum([ sc_logprob[s] + self.scenario_logprobs[s] for s in sc_logprob.keys()], axis = 0)

        # sort object IDs by their log probability in objectid_logprob,
        # largest first
        return np.array(self.object_ids)[objectid_logprob.argsort()][::-1]        
