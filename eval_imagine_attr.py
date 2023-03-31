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

from scipy import stats


import vgiterator
import sentence_util
from vgpaths import VGPaths, get_output_path
from vec_util import VectorInterface
from sds_imagine_util import  ImagineAttr


###
# settings



parser = ArgumentParser()
parser.add_argument('--outdir', help="directory to write output for inspection, default: inspect_output/imagine_att", default = "inspect_output/imagine_att/")
parser.add_argument('--vgdata', help="directory with VG data including frequent items, train/test split, topic model", default = "data/")
parser.add_argument('--trainperc', help = "percentage of object types to use for training PLSR, default 0.8", type = float, default = 0.8)
parser.add_argument('--num_att', help = "number of top attributes to use, default 500", type = int, default = 500)
parser.add_argument('--plsr_components', help = "number of components to use for PLSR, default 100", type = int, default = 100)
parser.add_argument('--num_inspect', help = "number of objects to write for inspection, default 20", type = int, default = 20)

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

###
# read vectors
vec_obj = VectorInterface(vgpath_obj)

available_objects = [o for o in vgobjects_attr_rel["objects"] if o in vec_obj.object_vec]
missing = len(vgobjects_attr_rel["objects"]) - len(available_objects)
if missing > 0:
    print("frequent objects without vectors:", missing, "out of", len(vgobjects_attr_rel["objects"]))

# randomly sample objects for the training part
training_objectlabels = random.sample(available_objects, int(args.trainperc * len(available_objects)))
test_objectlabels = [ o for o in available_objects if o not in training_objectlabels]

###
# count object/attribute frequencies
# as well as attribute frequencies in the training data, use the top k

split_zipfilename, split_filename = vgpath_obj.vg_traintest_zip_and_filename()
with zipfile.ZipFile(split_zipfilename) as azip:
    with azip.open(split_filename) as f:
        traintest_split = json.load(f)
    
trainset = set(traintest_split["train"])
testset = set(traintest_split["test"])

vgiter = vgiterator.VGIterator(vgobjects_attr_rel)

print("Training and applying PLSR")

attr_obj =  ImagineAttr(vgiter, vec_obj, training_objectlabels = training_objectlabels, img_ids = trainset, num_attributes = args.num_att, num_plsr_components = args.plsr_components)



# predictions for training objects
Ypredict_train, Ygoldtrain, training_used_object_labels = attr_obj.predict_forobj(training_objectlabels)

# prediction for test objects
Ypredict_test, Ygoldtest, test_used_object_labels = attr_obj.predict_forobj(test_objectlabels)


##
# write output for inspection:
# several objects from the test portion, with gold and predicted features
outpath = get_output_path(os.path.join(args.outdir, "eval_imagine_att_out.txt"))
inspect_objectindices = random.sample(list(range(len(test_used_object_labels))), args.num_inspect)
with open(outpath, "w") as outf:
    for objix in inspect_objectindices:
        objlabel = test_used_object_labels[objix]
        
        print("-----------\nObject:", objlabel, "\n", file = outf)
        
        print("Gold (top 20):", file = outf)
        for a, p in sorted(zip(attr_obj.attributelabels, attr_obj.obj_att_prob[objlabel]), key = lambda p:p[1], reverse = True)[:20]:
            if p > 0.0:
                print("\t", a, ":", p, file = outf)
        print(file = outf)
        
        print("Predicted (top 20):", file = outf)
        att_pred = zip(attr_obj.attributelabels, Ypredict_test[objix])
        for a, p in sorted(att_pred, key = lambda p:p[1], reverse = True)[:20]:
            print("\t", a, ":", p, file = outf)

#########3
print("Evaluating")

##
# first evaluation:
# overall quality of feature vectors for test objects:
# Spearman
def average_spearman(model_lists, gold_lists, single_list = False):
    if single_list:
        rhos = [ stats.spearmanr(model_lists, gl).statistic for gl in gold_lists]
    else:
        rhos = [ stats.spearmanr(mi, gi).statistic for mi, gi in zip(model_lists, gold_lists)]
        
    if len(rhos) > 0:
        return sum(rhos) / len(rhos)
    else:
        return 0.0

Ypredict_train = Ypredict_train.tolist()
Ypredict_test = Ypredict_test.tolist()

# frequency baseline weights
baseline_weights = [count for _, count in attr_obj.att_count.most_common(args.num_att)]

print("Average Spearman's rho for prediction on Training objects:", round(average_spearman(Ypredict_train, Ygoldtrain), 3),
      "on Test objects:", round(average_spearman(Ypredict_test, Ygoldtest), 3))
print("Average Spearman's rho, frequency baseline: Training objects:",
      round(average_spearman(baseline_weights, Ygoldtrain, single_list = True), 3),
      "Test objects:", 
      round(average_spearman(baseline_weights, Ygoldtest, single_list = True), 3))
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

        these_attlabels = [a for a in attlabels if a in attr_obj.attributelabels]
        if len(these_attlabels) > 0:
            for objlabel in test_imgid_obj[image_id][objref]:
                if objlabel in vec_obj.object_vec:
                    X_testsent.append(vec_obj.object_vec[objlabel])
                    goldatt_testsent.append(these_attlabels)
                    goldobj_testsent.append(objlabel)

# do the prediction
prediction_testsent =  attr_obj.predict(X_testsent)

# compute ranking of correct attribute in each prediction,
# separate into objects that went into the training for PLSR and
# objects that didn't

# atts: gold attributes
# attlist: list of all attributes
# attweights: weights of all attributes
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
    ranking_of_true_attrib.append( compute_att_ranking( goldatt_testsent[ ix], prediction_testsent[ix], attr_obj.attributelabels))
    base_ranking_of_true_attrib.append( compute_att_ranking( goldatt_testsent[ ix], baseline_weights, attr_obj.attributelabels))
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


        
        
    
