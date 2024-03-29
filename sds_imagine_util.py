## Katrin Erk Feb 2023
# Utilities for imagining objects from scenarios,
# and attributes from concepts,
# to be re-used across
# evaluation scripts and interactive SDS

import os
import sys
import json
import zipfile
import math
from collections import Counter
import numpy as np

from sklearn.cross_decomposition import PLSRegression

from vgnames import VGOBJECTS, VGATTRIBUTES, VGRELATIONS 


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
            if word.startswith(VGOBJECTS):
                wordid = vgindex_obj.o2ix(word[len(VGOBJECTS):])
                if wordid is None:
                    raise Exception("lookup error for topic word, object", word[len(VGOBJECTS):])

                self.object_ids.append(wordid)
                self.indices_of_objects.append(index)

        self.indices_of_objects = np.array(self.indices_of_objects)

        # for each scenario, restrict to objects, renormalize, store
        # as probabilities rather than log probabilities
        self.scenario_probs = [ ]
        for sclp in term_topic_matrix:

            a = np.exp(np.array(sclp))

            # select only columns that describe an object
            a = a[ self.indices_of_objects ]

            # renormalize
            normalizer = np.exp(a).sum()
            a = a / normalizer
            # scenario_probs: scenario probabilities for objects, in order of scenarios
            self.scenario_probs.append(a)


    def predict_objectids(self, scenario_list):
        # obtain scenario probabilities within the scenario list
        sc_freq = Counter(scenario_list)
        # last scenario is a dummy, it has no entry in scenario_probs
        dummyscenario = len(self.scenario_probs)
        del sc_freq[dummyscenario]
        
        # sc_prob: dictionary scenario ID -> probability of this scenario in this sentence
        normalizer = sum(sc_freq.values())
        sc_prob = dict( (s, sc_freq[s] / normalizer) for s in sc_freq.keys())

        
        # scenario_probs: list where the i-th entry is for the i-th scenario,
        #    each entry is a numpy array of object probabilities
        # compute: sum_{s scenario in this sentence} this_sentence_scenario_prob(s) * prob_of_each_object(s)
        # again a numpy array of object  probabilities
        objectid_prob = np.sum([ sc_prob[s] * self.scenario_probs[s] for s in sc_prob.keys()], axis = 0)
        objectid_prob = objectid_prob / sum(objectid_prob)

        # sort object IDs by their log probability in objectid_logprob,
        # largest first
        indices_smallest_to_largest = objectid_prob.argsort()
        sorted_object_ids = np.array(self.object_ids)[indices_smallest_to_largest][::-1]
        sorted_probs = objectid_prob[indices_smallest_to_largest][::-1] 

        return (sorted_object_ids, np.log(sorted_probs))

class ImagineAttr:
    def __init__(self, vgiter, vec_obj, training_objectlabels = None, img_ids = None, num_attributes = 500, num_plsr_components = 100):

        self.vec_obj = vec_obj

        # remember which object number (ref) goes with which object labels
        # image id -> objref -> object
        # both training and test, as we need gold info for test objects

        imgid_obj = { }
        for image_id, objref_objlabels in vgiter.each_image_objects_full(img_ids = img_ids):

            imgid_obj[ image_id] = { }

            for objref, objnames in objref_objlabels:
                imgid_obj[image_id][ objref ] = objnames


        # go through all images again,
        # count attributes, and attribute/object co-occurrences.
        self.att_count = Counter()
        obj_att_count = { }

        for image_id, attlabels_objref in vgiter.each_image_attributes_full(img_ids = img_ids):

            for attlabels, objref in attlabels_objref:

                # count occurrences of attributes
                self.att_count.update(attlabels)

                # count object-attribute co-occurrences
                if image_id not in imgid_obj or objref not in imgid_obj[image_id]:
                    print("could not find object in image, skipping", image_id, objref)
                    continue

                for objlabel in imgid_obj[image_id][objref]:
                    if objlabel not in obj_att_count: obj_att_count[objlabel] = Counter()
                    for attlabel in attlabels:
                        obj_att_count[objlabel][attlabel] += 1

        if training_objectlabels is None:
            training_objectlabels = list(obj_att_count.keys())
                

        ######
        # list of attributes to use,
        # and objecct/attribute probs to use in training
        self.attributelabels = [a for a, _ in self.att_count.most_common(num_attributes)]
        self.obj_att_prob = { }
        for obj in obj_att_count.keys():
            normalizer = sum([obj_att_count[obj][a] for a in self.attributelabels])
            if normalizer == 0:
                # print("no attribute counts for", obj)
                continue

            self.obj_att_prob[ obj ] = [ obj_att_count[obj][a] / normalizer for a in self.attributelabels]


        ####
        # train PLSR
        self.pls_obj = PLSRegression(n_components= num_plsr_components)
        Xtrain, Ytrain, self.used_training_objectlabels = self._make_pls_input(training_objectlabels)
        self.pls_obj.fit(Xtrain, Ytrain)

        

    ########
    # make a prediction from the trained PLSR model
    def predict_forobj(self, objectlabels):
        X, Y, used_objectlabels = self._make_pls_input(objectlabels)

        if len(used_objectlabels) == 0:
            return (None, None, [ ])

        prediction = self.pls_obj.predict(X)

        return (prediction, Y, used_objectlabels)

    def predict(self, X):
        return self.pls_obj.predict(X)
    
    # for the objects where we have vectors, make data structures
    # for input into Partial Least Squares regression
    def _make_pls_input(self, objectlabels):
        X = [ ]
        Y = [ ]
        used_object_labels  = [ ]

        for obj in objectlabels:
            if obj in self.vec_obj.object_vec and obj in self.obj_att_prob:
                X.append(self.vec_obj.object_vec[obj])
                Y.append( self.obj_att_prob[obj] )
                used_object_labels.append( obj )

        return (X, Y, used_object_labels)
