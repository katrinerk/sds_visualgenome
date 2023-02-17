# Katrin Erk January 2023
# 
# make train/test split of Visual Genome images
# report statistics


import sys
import os
import json
import zipfile
from collections import defaultdict, Counter
import random

import vgiterator
from vgpaths import VGPaths

vgpath_obj = VGPaths()


testpercentage = 0.1
random_seed = 78923456

##
# load list of images to use as data
vgcounts_zipfilename, vgcounts_filename = vgpath_obj.vg_counts_zip_and_filename()
with zipfile.ZipFile(vgcounts_zipfilename) as azip:
    with azip.open(vgcounts_filename) as f:
        vgitems = json.load(f)

        
##
# make random split
num_datapoints = len(vgitems["images"])
num_test_datapoints = int(num_datapoints * testpercentage)

random.seed(random_seed)
test_images = random.sample(vgitems["images"], num_test_datapoints)

##
# write output
testset = set(test_images)

out = { "test" : sorted(test_images),
        "train" : sorted([i for i in vgitems["images"] if i not in testset]) }

split_zipfilename, split_filename = vgpath_obj.vg_traintest_zip_and_filename()
with zipfile.ZipFile(split_zipfilename, "w", zipfile.ZIP_DEFLATED) as azip:
    azip.writestr(split_filename, json.dumps(out, indent = 1))
    

trainset = set(out["train"])
##
# compute statistics
freq_objects = set(vgitems["objects"])
freq_attributes = set(vgitems["attributes"])
freq_relations = set(vgitems["relations"])


train_objects= Counter()
test_objects = Counter()
train_attrib = Counter()
test_attrib = Counter()
train_rel = Counter()
test_rel = Counter()

vgobj = vgiterator.VGIterator()

# object counts
for img, frequent_objects in vgobj.each_image_objects():
    for label in frequent_objects:
        if img in trainset: train_objects[label] += 1
        elif img in testset: test_objects[label] += 1

# attribute counts
for img, frequent_attrib in vgobj.each_image_attributes():
    for label in frequent_attrib:
        if img in trainset: train_attrib[label] += 1
        elif img in testset: test_attrib[label] += 1

# relation counts
for img, frequent_rel in vgobj.each_image_relations():
    for label in frequent_rel:
        if img in trainset: train_rel[label] += 1
        elif img in testset: test_rel[label] += 1

print("--")
print("Number of object instances: Training:", train_objects.total())
#print(text_histogram.histogram(list(train_objects.values())))
print("Number of object instances: Test:", test_objects.total())
#print(text_histogram.histogram(list(test_objects.values())))
print("--")
print("Number of attribute instances: Training:", train_attrib.total())
#print(text_histogram.histogram(list(train_attrib.values())))
print("Number of attribute instances: Test", test_attrib.total())
#print(text_histogram.histogram(list(test_attrib.values())))
print("--")
print("Number of relation instances: Training:", train_rel.total())
#print(text_histogram.histogram(list(train_rel.values())))
print("Number of relation instances: Test", test_rel.total())
#print(text_histogram.histogram(list(test_rel.values())))
