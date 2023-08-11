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
from argparse import ArgumentParser

from nltk.corpus import wordnet
from sklearn.linear_model import LogisticRegression

from vgnames import VGOBJECTS, VGATTRIBUTES, VGRELATIONS 
import vgiterator
import sentence_util
from vgpaths import VGPaths, get_output_path
from vec_util import VectorInterface
from sds_imagine_util import  ImagineAttr
from hypernymy_util import HypernymHandler


parser = ArgumentParser()
parser.add_argument('--vgdata', help="directory with VG data including frequent items, train/test split, topic model", default = "data/")
parser.add_argument('--trainperc', help = "percentage of object types to use for training PLSR, default 0.8", type = float, default = 0.8)


args = parser.parse_args()



print("reading data")
vgpath_obj = VGPaths(vgdata = args.vgdata)

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

# object fmr handling hypernymy
hyper_obj = HypernymHandler(available_objects) 

    
# train/dev/test split
print("splitting objects into train/dev/test, and making classifiers")
hyper_obj.make_hypernym_classifiers(vec_obj, trainpercent = args.trainperc, random_seed = 9386)

###
# for each hypernym: train ridge regression classifier,
# evaluate
all_correct = { "dev" : 0, "test" : 0 }
all_dp = { "dev" : 0, "test" : 0 }
all_nodata = {"dev" : 0, "test" : 0}
detail_accuracy = defaultdict(dict)

for section in ["dev", "test"]:
    for hypernym, predicted, gold in hyper_obj.eval_predict_each(section, vec_obj):
        if predicted is None:
            # no classifier could be fit for this hypernym
            all_nodata[section] += 1
            continue


        # apply to development and test data
        this_accuracy = { }
        correct = sum([int(y == pred) for y, pred in zip(gold, predicted)])
        all_correct[ section ] += correct
        all_dp[ section ] += len(gold)

        detail_accuracy[hypernym][ section ] = correct / len(gold)
        
for hypernym, acc in detail_accuracy.items():
    print(hypernym, "dev accuracy", round(acc["dev"], 3), "test accuracy", round(acc["test"], 3))

print()
print("Number of hypernyms", len(hyper_obj.hypernym_names))
print(f"Missing training data for {all_nodata['dev']} hypernyms")
print("Number of items: training", len(hyper_obj.training_objectlabels), "dev", len(hyper_obj.dev_objectlabels), "test", len(hyper_obj.test_objectlabels))
print()

print("Overall accuracy, hypernyms with training data: dev", round(all_correct["dev"] / all_dp["dev"], 3),
      "test", round(all_correct["test"] / all_dp["dev"], 3))

if all_nodata["dev"] > 0:
    print("Overall accuracy, including no-data items: dev", round(all_correct["dev"] / all_dp["dev"] + all_nodata["dev"] * len(hyper_obj.dev_objectlabels), 3),
            "test", round(all_correct["test"] / all_dp["test"] + all_nodata["test"] * len(hyper_obj.test_objectlabels), 3))

