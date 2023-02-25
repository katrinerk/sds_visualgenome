# Katrin Erk January 2023
# evaluate SDS on enrichment/imagination task:
# imagine additional attributes for objects


import sys
import os
import json
import zipfile
from collections import defaultdict, Counter
import math
import numpy as np
import random
from argparse import ArgumentParser

from sklearn.cross_decomposition import PLSRegression
from scipy import stats

import vgiterator
import sentence_util
from vgindex import VgitemIndex
from vgpaths import VGPaths

# TO ADD: output for inspection

###
# settings



parser = ArgumentParser()
# parser.add_argument('--outdir', help="directory to write output for inspection, default: inspect_output/imagine_att", default = "inspect_output/imagine_att/")
parser.add_argument('--vgdata', help="directory with VG data including frequent items, train/test split, topic model", default = "data/")
parser.add_argument('--trainperc', help = "percentage of object types to use for training PLSR, default 0.8", type = float, default = 0.8)
parser.add_argument('--num_att', help = "number of top attributes to use, default 300", type = int, default = 300)
parser.add_argument('--plsr_components', help = "number of components to use for PLSR, default 20", type = int, default = 20)

args = parser.parse_args()


random.seed(6543)

####
# read data

print("Reading data")

vgpath_obj = VGPaths(vgdata = args.vgdata)

# most frequent objects, attributes, relations
vgcounts_zipfilename, vgcounts_filename = vgpath_obj.vg_counts_zip_and_filename()
with zipfile.ZipFile(vgcounts_zipfilename) as azip:
    with azip.open(vgcounts_filename) as f:
        vgobjects_attr_rel = json.load(f)


# randomly sample objects for the training part
training_objectlabels = random.sample(vgobjects_attr_rel["objects"], int(args.trainperc * len(vgobjects_attr_rel["objects"])))
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
testset = set(traintest_split["test"])

vgiter = vgiterator.VGIterator(vgobjects_attr_rel)

# remember which object number (ref) goes with which object labels
#image id -> objref -> object
# both training and test, as we need gold info for test objects

imgid_obj = { }
for image_id, objref_objlabels in vgiter.each_image_objects_full(img_ids = trainset):
    
    imgid_obj[ image_id] = { }
    
    for objref, objnames in objref_objlabels:
        imgid_obj[image_id][ objref ] = objnames


# go through all images again,
# count attributes, and attribute/object co-occurrences.


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
        


use_attributelabels = [a for a, _ in attname_count.most_common(args.num_att)]
obj_att_prob = { }
for obj in obj_att_count.keys():
    normalizer = sum([obj_att_count[obj][a] for a in use_attributelabels])
    if normalizer == 0:
        print("no attribute counts for", obj)
        continue
    
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


# # # do we have vectors for all objects?
# print("No entry for:")
# for obj in training_objectlabels:
#     if obj not in object_vec:
#         print(obj, end = ", ")
# print()


# for the objects where we have vectors, make data structures
# for input into Partial Least Squares regression
def make_pls_input(objectlabels, object_vec, object_att_prob):

    X = [ ]
    Y = [ ]
    used_object_labels  = [ ]
    
    for obj in objectlabels:
        if obj in object_vec:
            X.append(object_vec[obj])
            Y.append( object_att_prob[obj] )
            used_object_labels.append( obj )

    return (X, Y, used_object_labels)

print("Training and applying PLSR")

Xtrain, Ytrain, training_used_object_labels = make_pls_input(training_objectlabels, object_vec, obj_att_prob)
Xtest, Ytest, test_used_object_labels = make_pls_input(test_objectlabels, object_vec, obj_att_prob)

# train PLS regression
pls_obj = PLSRegression(n_components=args.plsr_components)
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
def average_spearman(model_lists, gold_lists, single_gold_list = False):
    if single_gold_list:
        rhos = [ stats.spearmanr(mi, gold_lists).statistic for mi in model_lists]
    else:
        rhos = [ stats.spearmanr(mi, gi).statistic for mi, gi in zip(model_lists, gold_lists)]
        
    if len(rhos) > 0:
        return sum(rhos) / len(rhos)
    else:
        return 0.0

Ypredict_train = Ypredict_train.tolist()
Ypredict_test = Ypredict_test.tolist()

# frequency baseline weights
baseline_weights = [count for _, count in attname_count.most_common(args.num_att)]

print("Average Spearman's rho for prediction on Training objects:", round(average_spearman(Ypredict_train, Ytrain), 3),
      "on Test objects:", round(average_spearman(Ypredict_test, Ytest), 3))
print("Average Spearman's rho, frequency baseline: Training objects:",
      round(average_spearman(Ypredict_train, baseline_weights, single_gold_list = True), 3),
      "Test objects:", 
      round(average_spearman(Ypredict_test, baseline_weights, single_gold_list = True), 3))
##
# second evaluation:
# for individual sentences from the test data,
# predict withheld attributes.
# evaluate separately for objects seen and unseen
# during fitting the PLSR model

##
# read objects and attributes in test sentences
test_imgid_obj = { }
for image_id, objref_objlabels in vgiter.each_image_objects_full(img_ids = testset):
    
    test_imgid_obj[ image_id] = { }
    
    for objref, objnames in objref_objlabels:
        test_imgid_obj[image_id][ objref ] = objnames

# for each object with attribute in the test sentences:
# store object vector, gold attribute
X_testsent = [ ]
goldatt_testsent = [ ]
goldobj_testsent = [ ]

for image_id, attlabels_objref in vgiter.each_image_attributes_full(img_ids = testset):

    for attlabels, objref in attlabels_objref:

        # count object-attribute co-occurrences
        if image_id not in test_imgid_obj or objref not in test_imgid_obj[image_id]:
            print("could not find object in image, skipping", image_id, objref)
            continue

        these_attlabels = [a for a in attlabels if a in use_attributelabels]
        if len(these_attlabels) > 0:
            for objlabel in test_imgid_obj[image_id][objref]:
                if objlabel in object_vec:
                    X_testsent.append(object_vec[objlabel])
                    goldatt_testsent.append(these_attlabels)
                    goldobj_testsent.append(objlabel)

# do the prediction
prediction_testsent =  pls_obj.predict(X_testsent)

# compute ranking of correct attribute in each prediction,
# separate into objects that went into the training for PLSR and
# objects that didn't

def compute_att_ranking(atts, attweights, attlist):
    # compute a ranking of items in attweights
    # in a_ranking, the entries are the ranks of items from 0 to n

    # first, compute order of weights using argsort
    a_order = np.array(attweights).argsort()
    # using argsort twice gives us a ranking of weights.
    # we invert the order to rank from highest to lowest weight
    a_ranking = a_order[::-1].argsort()

    # compute position in ranking for each attribute of the current noun
    ranks = [ ]
    for att in atts:
        attix = attlist.index(att)
        ranks.append( a_ranking [ attix ])

    # and average
    return sum(ranks) / len(ranks)

    
        

ranking_of_true_attrib = [ ]
base_ranking_of_true_attrib = [ ]
obj_is_train = [ ]

for ix, obj in enumerate(goldobj_testsent):
    ranking_of_true_attrib.append( compute_att_ranking( goldatt_testsent[ ix], prediction_testsent[ix], use_attributelabels))
    base_ranking_of_true_attrib.append( compute_att_ranking( goldatt_testsent[ ix], baseline_weights, use_attributelabels))
    obj_is_train.append( int(obj in training_objectlabels))
                                            

print("Predicting attributes for objects occurring in test sentences:")
print("Average rank of true attribute in prediction ranking:")

vals = [r for ix, r in enumerate(ranking_of_true_attrib) if obj_is_train[ix]]
print("Model prediction, training objects:", round(sum(vals) / len(vals), 3))

vals = [r for ix, r in enumerate(base_ranking_of_true_attrib) if obj_is_train[ix]]
print("Baseline, training objects:", round(sum(vals) / len(vals), 3))

vals = [r for ix, r in enumerate(ranking_of_true_attrib) if not obj_is_train[ix]]
print("Model prediction, test objects:", round(sum(vals) / len(vals), 3))

vals = [r for ix, r in enumerate(base_ranking_of_true_attrib) if not obj_is_train[ix]]
print("Baseline, test objects:", round(sum(vals) / len(vals), 3))


