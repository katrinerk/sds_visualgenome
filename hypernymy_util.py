## Katrin Erk August 2023
# Utilities for handling hypernyms

import sys
import random
import nltk
import math
from collections import defaultdict, Counter
from nltk.corpus import wordnet
from sklearn.linear_model import LogisticRegression

from vec_util import VectorInterface


class HypernymHandler:
    def __init__(self, vgobjects, index_of_first_hypernym = 0):
        # remember index of firt hypernym
        self.index_of_first_hypernym = index_of_first_hypernym

        # cutoffs for medium-frequency hypernyms
        self.hypernym_frequency_cutoff = (10, 700)

        # compute hypernyms for given object labels
        self.hypernym_names, self.object_hypernyms = self._determine_object_hypernyms(vgobjects)
        self.object_names = sorted(list(self.object_hypernyms.keys()))

        # empty initializations for train/dev/test objects, classifiers
        self.training_objectlabels = None
        self.dev_objectlabels = None
        self.test_objectlabels = None
        self.classifiers =  None
        

    ##
    # map hypernym label to an index, taking into account the index of first hypernym
    # given on initialization
    #
    # returns: integer index, or None
    def hypernym_index(self, hypernym_name):
        try:
            # this may fail if the hypernym is not in there
            hypernym_offset = self.hypernym_names.index(hypernym_name)
            return self.index_of_first_hypernym + hypernym_offset
        except ValueError:
            return None

    def hypernym_ixlabel(self, hypernym_index):
        offset = hypernym_index - self.index_of_first_hypernym
        return self.hypernym_names[offset]
        
    ##
    # classifiers for hypernyms:
    # making a train/dev/test split on objects for which we have hypernyms,
    # and training classifiers using the training portion of object labels
    #
    # returns: 3 lists of object labels: train, dev, test
    def make_hypernym_classifiers(self, vec_obj, trainpercent = 0.8, random_seed = 0):
        
        # make train/dev/test split
        random.seed(random_seed)
        
        self.training_objectlabels = random.sample(self.object_names, int(trainpercent * len(self.object_names)))
        nontraining_objectlabels = [ o for o in self.object_names if o not in self.training_objectlabels]
        self.dev_objectlabels = random.sample(nontraining_objectlabels, int(0.5 * len(nontraining_objectlabels)))
        self.test_objectlabels = [o for o in nontraining_objectlabels if o not in self.dev_objectlabels]

        # make matrix of vectors for training objects
        Xtrain = [  vec_obj.object_vec[label] for label in self.training_objectlabels ]

        self.classifiers =  { }
        
        for hypernym in self.hypernym_names:
            postraininglabels = [o for o in self.object_names if hypernym in self.object_hypernyms[o]]
            if len(postraininglabels) == 0:
                # no training data for this hypernym
                continue 

            ytrain = [ int(ell in postraininglabels) for ell in self.training_objectlabels]
            self.classifiers[hypernym] = LogisticRegression(random_state=0)
            self.classifiers[hypernym].fit(Xtrain, ytrain)
            


    ##
    # for the sake of evaluation, apply hypernymy classifiers
    # to all dev or all test data
    def eval_predict_each(self, section, vec_obj):
        if section == "dev":
            section_objectlabels = self.dev_objectlabels
        elif section == "test":
            section_objectlabels = self.test_objectlabels
        else:
            raise Exception("section needs to be dev or test, I got " + str(section))

        X = [  vec_obj.object_vec[label] for label in section_objectlabels ]

        for hypernym in self.hypernym_names:
            if hypernym not in self.classifiers:
                # couldnt' train a classifier for this one
                yield (hypernym, None, None)

            else:
                # we do have a classifier, use it
                gold_hyponyms =  [o for o in section_objectlabels if hypernym in self.object_hypernyms[o]]
                yield( hypernym, self.classifiers[hypernym].predict(X), [int(ell in gold_hyponyms) for ell in section_objectlabels])


    ##
    # adapt parameters for the SDS model to the presence of hypernyms
    def adapt_sds_params(self, global_param, scenario_concept_param, word_concept_param, selpref_param, vgindex_obj):
        ## global parameters:
        # add a concept for each hypernym, and a matching word
        num_hyper = len(self.hypernym_names)
        global_param["num_words"] += num_hyper
        global_param["num_concepts"] += num_hyper

        ## scenario/concept weights:
        # scenario/hypernym weight is average of scenario/concept weights
        # for hyponym concepts
        for h in self.hypernym_names:
            h_id = self.hypernym_index(h)
            hypos = [vgindex_obj.o2ix(o) for o in self.object_names if h in self.object_hypernyms[o]]
            if len(hypos) == 0:
                raise Exception("shouldn't be here " + h)

            # entry for the concept that is the hypernym ID:
            # "scenario": list of all scenario IDs,
            # "weight": average weight across all hyponym IDs for this scenario
            scenario_concept_param[ h_id ] = { "scenario" : [], "weight" : [ ]}
            for scenario_no in range(global_param["num_scenarios"] - 1):
                # weights are log probabilities.
                # so do sum of exp's of weights that all hyponyms have for this scenario
                weights = [ ]
                for hoid in hypos:
                    if scenario_no in scenario_concept_param[ hoid ]["scenario"]:
                        scix = scenario_concept_param[ hoid ]["scenario"].index(scenario_no)
                        weights.append( scenario_concept_param[ hoid ]["weight"][scix])

                if len(weights) > 0:
                    summed_weight = sum([math.exp(w) for w in weights])
                    # average, and log, and store
                    scenario_concept_param[ h_id ]["scenario"].append(scenario_no)
                    scenario_concept_param[ h_id ]["weight"].append( math.log(summed_weight / len(weights)) )

        ## word/concept weights:
        # nothing to add, a hypernym word only maps to the hypernym concept

        ## selectional preference weights:
        # average of concept-specific preferences
        # for hyponym concepts
        
        # for each predicate index, collect weights of hyponyms
        for h in self.hypernym_names:
            h_id = self.hypernym_index(h)
            hypos = [vgindex_obj.o2ix(o) for o in self.object_names if h in self.object_hypernyms[o]]
            if len(hypos) == 0:
                raise Exception("shouldn't be here " + h)

            # compute separately for arg0 and arg1
            for argpos in ["arg0", "arg1"]:
                # mapping predicate index -> list of log weights of hyponyms of h
                predix_weights = defaultdict(list)
                # iterate through predix/argix pairs for this arg position
                for index, entry in enumerate(selpref_param[ argpos ]["config"]):
                    predix, argix = entry
                    # the argument is one of the hyponyms: store the corresponding weight
                    if argix in hypos: predix_weights[ predix ].append( selpref_param[ argpos ]["weight"][index] )

                # for each predicate index, store the average weight for the hypernym
                for predix in predix_weights:
                    selpref_param[ argpos ]["config"].append( (predix, h_id) )
                    selpref_param[ argpos ]["weight"].append( math.log( sum([math.exp(w) for w in predix_weights[predix ]]) / len(hypos) ) )
            

        
        return (global_param, scenario_concept_param, word_concept_param, selpref_param)


    #####
    # compute additional parameters for hypernymy handling in SDS
    #
    # returns:
    # hypernymy parameters for SDS, format
    # { "concept-hyper" : 
    #   { concept index : {"hyper" : list of hypernym concept indices,
    #                      "weight" : list of log weights matching the hypernym concept indices }},
    #  "hyper-hyper" : SOMETHING }
    #
    # plus list of indices of training object labels
    def compute_hyper_param(self, global_param, vgindex_obj, vec_obj, condition = "ch", random_seed = 0):
        if condition != "ch":
            raise Exception("can't handle other conditions than ch yet")

        # train hypernymy classifiers.
        # this also sets self.training_objectlabels,
        #   self.dev_objectlabels ,self.test_objectlabels.
        # We want to return training object labels too.
        self.make_hypernym_classifiers(vec_obj, random_seed = random_seed)

        # get log prob predictions, for each hypernym, for all object labels
        retv = { "concept-hyper" : { }}

        # basis for prediction: vectors
        X = [  vec_obj.object_vec[label] for label in self.object_names ]
        # object IDs
        object_ids = [vgindex_obj.o2ix(ell) for ell in self.object_names]

        for hypernym in self.hypernym_names:
            if hypernym not in self.classifiers:
                # couldnt' train a classifier for this one
                continue

            hyperid = self.hypernym_index(hypernym)

            # we do have a classifier, use it
            predictions = self.classifiers[hypernym].predict_log_proba(X)

            # and store the results
            for index, objid in enumerate(object_ids):
                if objid not in retv["concept-hyper"]: retv["concept-hyper"][objid] = {"hyper" : [], "weight":[]}
                retv["concept-hyper"][objid]["hyper"].append(hyperid)
                retv["concept-hyper"][objid]["weight"].append( predictions[ index ] )

        return (retv, [vgindex_obj.o2ix(o) for o in self.training_objectlabels])

          
        
    ################
    # main preprocessing step:
    # for the given set of frequent object labels,
    # determine all wordnet hypernyms
    # that aren't too frequent or too rare,
    # and that have names that are themselves frequent in English
    def _determine_object_hypernyms(self, objectlabels):

        # map object labels to hypernym synsets and vice versa
        word_hyper = nltk.ConditionalFreqDist()
        hyper_word = nltk.ConditionalFreqDist()


        for objectlabel in objectlabels:
            hyps = self._all_hypernyms_of(objectlabel)

            for h in hyps:
                word_hyper[objectlabel][h] +=1
                hyper_word[h][objectlabel] += 1

        # restrict to medium-frequency hypernyms
        # that have names that are themselves frequent words
        # checking against a list of common words
        # google-10000-english-usa.txt
        # obtained from:
        # https://github.com/first20hours/google-10000-english/blob/master/google-10000-english-usa.txt
        # this is:
        # "This repo contains a list of the 10,000 most common English words in order of frequency,
        # as determined by n-gram frequency analysis of the Google's Trillion Word Corpus."
        lowerlimit, upperlimit = self.hypernym_frequency_cutoff
        overlycommon_hyp = set([h for h in hyper_word.keys() if hyper_word[h].N() > upperlimit])
        rare_hyp = set([h for h in hyper_word.keys() if hyper_word[h].N() <= lowerlimit])

        with open("google-10000-english-usa.txt") as f:
            frequentwords = [w.strip() for w in  f.readlines()]

        good_hyp = set()
        for h in hyper_word.keys():
            if h in overlycommon_hyp or h in rare_hyp: continue
        
            lemmas = [l.name() for l in h.lemmas()]
            if any( l in frequentwords for l in lemmas):
                good_hyp.add(h)


        # make a mapping from object labels to their "good" hypernyms, as strings
        word_goodhyper = { }
        for objectlabel in word_hyper.keys():
            this_hyper = [h.name() for h in word_hyper[objectlabel] if h in good_hyp]
            if len(this_hyper) > 0:
                word_goodhyper[ objectlabel ] = this_hyper

        # make list of good hypernyms as strings
        good_hypstring = [h.name() for h in hyper_word.keys() if h in good_hyp]

        return (good_hypstring, word_goodhyper)



    def _all_hypernyms_of(self, objectlabel):
        # determine synsets
        synsets = wordnet.synsets(objectlabel, pos = wordnet.NOUN)
        if len(synsets) == 0:
            objectlabel = objectlabel.split()[-1]
            synsets = wordnet.synsets(objectlabel, pos = wordnet.NOUN)

            # no synsets found
            if len(synsets) == 0:
                return [ ]

        hyps = set()
        hypfn = lambda s:s.hypernyms()

        for syn0 in synsets:

            # hypernyms
            hyper_synsets = list(syn0.closure(hypfn))
            hyps.update(hyper_synsets)

        return list(hyps)
