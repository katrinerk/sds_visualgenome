# Katrin Erk January 2023
# run Latent Dirichlet Allocation, using gensim, on Visual Genome images,
# store probabilities p(word | topic)

import sys
import os
import zipfile
import json
import pickle
from collections import defaultdict, Counter
import nltk
import math
import gensim
import numpy as np

from vgpaths import VGPaths

vgpath_obj = VGPaths()


# let Gensim determine best alpha value for each topic
dirichlet_alpha = 0.05
alphatype = "symmetric"
# number of scenarios
num_topics = 20

print("reading corpus")        
# Gensim dictionary file
gensimdict_zipfilename, gensimdict_filename = vgpath_obj.gensim_dict_zip_and_filename()
with zipfile.ZipFile(gensimdict_zipfilename) as azip:
    with azip.open(gensimdict_filename) as f:
        gensim_dictionary = pickle.load(f)

# Gensim corpus files
corpus = [ ]
zip_corpusfilename = vgpath_obj.gensim_corpus_zipfilename()
with zipfile.ZipFile(zip_corpusfilename) as azip:
    for filename in azip.namelist():
        with azip.open(filename) as f:
            corpus.append(json.load(f))

          
print("training LDA model")
ldamodel = gensim.models.ldamodel.LdaModel(corpus,
                                           num_topics = num_topics,
                                           id2word=gensim_dictionary, 
                                           alpha = dirichlet_alpha,
                                           passes = 10, iterations = 500, 
                                           random_state = 2, per_word_topics = True)


##
# write:
gensim_zipfilename, overall_filename, topic_filename, word_filename, topicword_filename = vgpath_obj.gensim_out_zip_and_filenames()
with zipfile.ZipFile(gensim_zipfilename, "w", zipfile.ZIP_DEFLATED) as azip:
    # Dirichlet alpha values, number of topics
    azip.writestr(overall_filename, json.dumps( {"alpha" : dirichlet_alpha if alphatype == "symmetric" else ldamodel.alpha.tolist(),
                                               "alphatype" : alphatype, "num_topics" : num_topics}))
    
    # overall characterization of each topic in terms of its 10 most likely words
    topicinfo = []
    for topicnumber in range(num_topics):
        topicinfo.append( {"topic" : topicnumber, "descr" : [ [word, prob.item()] for word, prob in ldamodel.show_topic(topicnumber)]})
    azip.writestr(topic_filename, json.dumps(topicinfo))
    
    # ordered list of words that appear in topics
    numwords = max(gensim_dictionary.token2id.values()) + 1
    azip.writestr(word_filename, json.dumps([gensim_dictionary.get(i, "NONE") for i in range(numwords)]))

    # term-document matrix, one row per topic, one column per word
    # use log probabilities rather than probabilities
    azip.writestr(topicword_filename, json.dumps(np.log(ldamodel.get_topics()).tolist()))
    
        
# for wordix in range(10):
#     print(gensim_dictionary[wordix], ldamodel.get_term_topics(wordix))

# termtopics = ldamodel.get_topics()

# for wordix in range(10):
#     print(termtopics[:, wordix])

# print(ldamodel.alpha)
