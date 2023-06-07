# Katrin Erk June 2023
# hypernyms of objects in the Visual Genome


import sys
import os
import json
import zipfile
from collections import defaultdict, Counter
import math
import statistics
import random
import nltk
from nltk.corpus import wordnet

from vgnames import VGOBJECTS, VGATTRIBUTES, VGRELATIONS 
import vgiterator
import sentence_util
from vgpaths import VGPaths, get_output_path
from vec_util import VectorInterface
from sds_imagine_util import  ImagineAttr


trainperc = 0.8

print("reading data")
# vgpath_obj = VGPaths(vgdata = args.vgdata)
vgpath_obj = VGPaths()

# most frequent objects, attributes, relations
vgcounts_zipfilename, vgcounts_filename = vgpath_obj.vg_counts_zip_and_filename()
with zipfile.ZipFile(vgcounts_zipfilename) as azip:
    with azip.open(vgcounts_filename) as f:
        vgobjects_attr_rel = json.load(f)

vec_obj = VectorInterface(vgpath_obj)

available_objects = [o for o in vgobjects_attr_rel[VGOBJECTS] if o in vec_obj.object_vec]
missing = len(vgobjects_attr_rel[VGOBJECTS]) - len(available_objects)
if missing > 0:
    print("frequent objects without vectors:", missing, "out of", len(vgobjects_attr_rel[VGOBJECTS]))

# train/dev/test split
print("splitting objects into train/dev/test")
random.seed(9386)
training_objectlabels = random.sample(available_objects, int(trainperc * len(available_objects)))
nontraining_objectlabels = [ o for o in available_objects if o not in training_objectlabels]
dev_objectlabels = random.sample(nontraining_objectlabels, int(0.5 * len(nontraining_objectlabels)))
test_objectlabels = [o for o in nontraining_objectlabels if o not in dev_objectlabels]

###
print("determining hypernyms")

hypernym_candidates = Counter()
hypfn = lambda s:s.hypernyms()

for objectlabel in training_objectlabels:

    synsets = wordnet.synsets(objectlabel, pos = wordnet.NOUN)
    # did we find any synsets?
    if len(synsets) == 0:
        continue

    # first synset
    syn0 = synsets[0]

    # hypernyms
    hyper_synsets = syn0.closure(hypfn)

    # only retain hypernyms that contain a one-word lemma
    hyper_synsets_retained = [ ]
    for h in hyper_synsets:
        lemmas = [l.name() for l in h.lemmas()]
        lemmas = [l for l in lemmas if "_" not in l]
        if len(lemmas) > 0:
            hyper_synsets_retained.append(h)
    
    # count these hypernyms
    for h in hyper_synsets_retained:
        hypernym_candidates[ h.name() ] += 1

numhyp = 0
for hyper, count in hypernym_candidates.most_common():
    if count >= 140 or count < 10:
        continue
    numhyp += 1
    print(hyper, wordnet.synset(hyper).lemmas(), count)

print(numhyp)
