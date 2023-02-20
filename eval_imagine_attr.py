# Katrin Erk January 2023
# evaluate SDS on enrichment/imagination task:
# imagine additional attributes for objects


import sys
import os
import json
import zipfile
from collections import defaultdict, Counter
import nltk
import math
import numpy as np
import random
import statistics
import csv

from sklearn.cross_decomposition import PLSRegression

import vgiterator
import sentence_util
from vgindex import VgitemIndex
from vgpaths import VGPaths

###
# settings

# what percentage of objects to use for training?
percentage_training_objects = 0.8

# use top how many attributes?
top_n_attributes = 300

pls_num_components = 20

# write readable output about a random sample of
# how many sentences?
num_sentences_to_inspect = 10

outdir = "inspect_output/imagine_attr"

random.seed(6543)

####
# read data

print("Reading data")


vgpath_obj = VGPaths()

# most frequent objects, attributes, relations
vgcounts_zipfilename, vgcounts_filename = vgpath_obj.vg_counts_zip_and_filename()
with zipfile.ZipFile(vgcounts_zipfilename) as azip:
    with azip.open(vgcounts_filename) as f:
        vgobjects_attr_rel = json.load(f)


# randomly sample objects for the training part
training_objectlabels = random.sample(vgobjects_attr_rel["objects"], int(percentage_training_objects * len(vgobjects_attr_rel["objects"])))
test_objectlabels = [ o for o in vgobjects_attr_rel["objects"] if o not in training_objectlabels]

###
# count object/attribute frequencies
# as well as attribute frequencies in the training data, use the top k

print("Obtaining frequencies")

split_zipfilename, split_filename = vgpath_obj.vg_traintest_zip_and_filename()
with zipfile.ZipFile(split_zipfilename) as azip:
    with azip.open(split_filename) as f:
        traintest_split = json.load(f)
    
trainset = set(traintest_split["train"])

vgiter = vgiterator.VGIterator(vgobjects_attr_rel)

# remember which object number (ref) goes with which object labels
#image id -> objref -> object
imgid_obj = { }
for image_id, objref_objlabels in vgiter.each_image_objects_full(img_ids = trainset):
    
    imgid_obj[ image_id] = { }
    
    for objref, objnames in objref_objlabels:
        imgid_obj[ objref ] = objnames

# go through all images again,
# count attributes, and attribute/object co-occurrences

attname_count = Counter()
obj_att_count = { }

for image_id, attlabels_objref in vgiter.each_image_attributes_full(img_ids = trainset):

    for attlabels, objref in attlabels_objref:
        
        # count occurrences of attributes
        attname_count.update(attlabels)

        # count object-attribute co-occurrences
        if image_id not in imgid_obj or objref not in imgid_obj[image_id]:
            print("could not find object in image, skipping", image_id, objref)
            continue

        for objlabel in imgid_obj[image_id][objref]:
            if objlabel not in obj_att_count: obj_att_count[objlabel] = Counter()
            for attlabel in attlabels:
                obj_att_count[objlabel][attlabel] += 1
        


use_attributelabels = [a for a, _ in attname_count.most_common(top_n_attributes)]
obj_att_prob = { }
for obj in obj_att_count.keys():
    normalizer = sum([obj_att_count[obj][a] for a in use_attributelabels])
    obj_att_prob[ obj ] = [ obj_att_count[obj][a] / normalizer for a in use_attributelabels]

###
# read vectors
vecfilename = vgpath_obj.vg_vecfilename()
object_vec = { }

with open(vecfilename) as f:
    for line in f:
        pieces = line.split()
        # first word is label
        label = pieces[0]
        # from what I can see, vectors for attributes have no .n at the end,
        # and vectors for relations are actually for relation-argument pairs,
        # and always have parentheses.
        # what are they for?
        #
        # determine vectors for objects:
        # remove .n
        # and turn underscores to spaces
        if label.endswith(".n") and "(" not in label and ")" not in label:
            label = label[:-2]
            label = label.replace("_", " ")
            object_vec[ label ] = [ float(v) for v in pieces[1:]]


# # do we have vectors for all objects?
print("No entry for:")
for obj in training_objectlabels:
    if obj not in object_vec:
        print(obj, end = ", ")
print()


# for the objects where we have vectors, make data structures
# for input into Partial Least Squares regression
def make_pls_input(objectlabels, object_vec, object_att_prob):

    X = [ ]
    Y = [ ]
    used_object_labels  [ ]
    
    for obj in objectlabels:
    if obj in object_vec:
        X.append(object_vec[obj])
        Y.append( object_att_prob [obj] )
        used_object_labels.append( obj )

    return (X, Y, used_object_labels)

print("Training and applying PLSR")

Xtrain, Ytrain, training_used_object_labels = make_pls_input(training_objectlabels, object_vec, obj_att_prob)
Xtest, Ytest, test_used_object_labels = make_pls_input(test_objectlabels, object_vec, obj_att_prob)

# train PLS regression
pls_obj = PLSRegression(n_components=pls_num_components)
pls_obj.fit(Xtrain, Ytrain)

# predictions for training objects
Ypredict_train = pls_obj.predict(Xtrain)

# prediction for test objects
Ypredict_test = pls_obj.predict(Xtest)

#########3
print("Evaluating")

##
# first evaluation:
# overall quality of feature vectors for test objects:
# Spearman
