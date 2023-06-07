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

# train/dev/test split
print("splitting objects into train/dev/test")
random.seed(9386)
training_objectlabels = random.sample(available_objects, int(args.trainperc * len(available_objects)))
nontraining_objectlabels = [ o for o in available_objects if o not in training_objectlabels]
dev_objectlabels = random.sample(nontraining_objectlabels, int(0.5 * len(nontraining_objectlabels)))
test_objectlabels = [o for o in nontraining_objectlabels if o not in dev_objectlabels]

###
# hypernyms that we're using
hypernym_names_str = """person.n.01
covering.n.02
device.n.01 
structure.n.01
thing.n.12 
container.n.01
substance.n.01
commodity.n.01
animal.n.01 
solid.n.01 
clothing.n.01
vertebrate.n.01
chordate.n.01 
food.n.02 
substance.n.07
material.n.01 
conveyance.n.03 
vehicle.n.01 
vascular_plant.n.01 
plant.n.02 
shape.n.02 
placental.n.01
mammal.n.01
implement.n.01
garment.n.01 
produce.n.01 
decoration.n.01
foodstuff.n.02 
furnishing.n.02
creation.n.02 
equipment.n.01
protective_covering.n.01
signal.n.01
ungulate.n.01
vegetable.n.01
extremity.n.04
furniture.n.01 
craft.n.02 
nutriment.n.01
vessel.n.03 
surface.n.01
herb.n.01
fruit.n.01"""

hypernym_names = hypernym_names_str.split()

###
# for each training object label: what are its hypernyms?
def objectlabels_determine_hypernyms(objectlabels, hypernym_names):
    # mapping hypernym name -> list of object labels
    retv = defaultdict(list)
    
    hypfn = lambda s:s.hypernyms()
    hypset = set(hypernym_names)

    for objectlabel in objectlabels:
        synsets = wordnet.synsets(objectlabel, pos = wordnet.NOUN)
        if len(synsets) == 0:
            continue
        
        syn0 = synsets[0]
        thisobj_hypernyms = [s.name() for s in syn0.closure(hypfn)]

        for s in thisobj_hypernyms:
            if s in hypset:
                retv[ s].append(objectlabel)

    return retv

training_obj_hyponyms = objectlabels_determine_hypernyms(training_objectlabels, hypernym_names)
dev_obj_hyponyms = objectlabels_determine_hypernyms(dev_objectlabels, hypernym_names)
test_obj_hyponyms = objectlabels_determine_hypernyms(test_objectlabels, hypernym_names)

###
# make matrix of vectors for training objects, for dev objects
Xtrain = [  vec_obj.object_vec[label] for label in training_objectlabels ]
Xdev = [  vec_obj.object_vec[label] for label in dev_objectlabels ]
Xtest = [  vec_obj.object_vec[label] for label in test_objectlabels ]

###
# for each hypernym: train ridge regression classifier,
# evaluate
all_correct = { "dev" : 0, "test" : 0 }
all_dp = { "dev" : 0, "test" : 0 }
all_nodata = 0

for hypernym in hypernym_names:
    # do we have training data for this hypernym?
    if hypernym not in training_obj_hyponyms:
        # no training data for this hypernym
        all_nodata += 1
        continue

    # obtaining positive training labels for this hypernym,
    # making data, training a classifier
    postraininglabels = training_obj_hyponyms[ hypernym ]
    ytrain = [ int(ell in postraininglabels) for ell in training_objectlabels]
    cl_obj = LogisticRegression(random_state=0)
    cl_obj.fit(Xtrain, ytrain)

    # apply to development and test data
    this_accuracy = { }
    for section, X, obj_hyponyms, objectlabels in [ ("dev", Xdev, dev_obj_hyponyms, dev_objectlabels),
                                                    ("test", Xtest, test_obj_hyponyms, test_objectlabels) ]:
        # make predictions
        predicted = cl_obj.predict(X)

        # evaluate
        yvals = [int(ell in obj_hyponyms[hypernym]) for ell in objectlabels]
        correct = sum([int(y == pred) for y, pred in zip(yvals, predicted)])
        all_correct[ section ] += correct
        all_dp[ section ] += len(yvals)

        this_accuracy[ section ] = correct / len(yvals)
        
    print(hypernym, "dev accuracy", round(this_accuracy["dev"], 3), "test accuracy", round(this_accuracy["test"], 3))

print()
print("Number of hypernyms", len(hypernym_names))
print(f"Missing training data for {all_nodata} hypernyms")
print("Number of items: training", len(training_objectlabels), "dev", len(dev_objectlabels), "test", len(test_objectlabels))
print()

print("Overall accuracy, hypernyms with training data: dev", round(all_correct["dev"] / all_dp["dev"], 3),
      "test", round(all_correct["test"] / all_dp["dev"], 3))

print("Overall accuracy, including no-data items: dev", round(all_correct["dev"] / all_dp["dev"] + all_nodata * len(dev_objectlabels), 3),
        "test", round(all_correct["test"] / all_dp["test"] + all_nodata * len(dev_objectlabels), 3))

