## Katrin Erk August 2023
# Utilities for handling hypernyms

import sys
import random
import nltk
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
        
    ##
    # classifiers for hypernyms:
    # making a train/dev/test split on objects for which we have hypernyms,
    # and training classifiers using the training portion of object labels
    #
    # returns: 3 lists of object labels: train, dev, test
    def make_hypernym_classifiers(self, vec_obj, trainpercent = 0.8, random_seed = 0):
        
        # make train/dev/test split
        random.seed(random_seed)
        
        available_objects = self.object_hypernyms.keys()
        
        self.training_objectlabels = random.sample(available_objects, int(trainpercent * len(available_objects)))
        nontraining_objectlabels = [ o for o in available_objects if o not in self.training_objectlabels]
        self.dev_objectlabels = random.sample(nontraining_objectlabels, int(0.5 * len(nontraining_objectlabels)))
        self.test_objectlabels = [o for o in nontraining_objectlabels if o not in self.dev_objectlabels]

        # make matrix of vectors for training objects
        Xtrain = [  vec_obj.object_vec[label] for label in self.training_objectlabels ]

        self.classifiers =  { }
        
        for hypernym in self.hypernym_names:
            postraininglabels = [o for o in available_objects if hypernym in self.object_hypernyms[o]]
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
