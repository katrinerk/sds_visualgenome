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
import configparser
from argparse import ArgumentParser


import vgiterator
from vgpaths import VGPaths

# where to write the output
parser = ArgumentParser()
parser.add_argument('--vgdata', help="directory with VG frequent-item information and train/test split", default = "data/")
args = parser.parse_args()

vgpath_obj = VGPaths(vgdata = args.vgdata)


# what percentage of data is for the test set?
config = configparser.ConfigParser()
config.read("settings.txt")

devtestpercentage = float(config["Parameters"]["Testpercentage"])

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
num_devtest_datapoints = int(num_datapoints * devtestpercentage)

random.seed(random_seed)
devtest_images = random.sample(vgitems["images"], num_devtest_datapoints)

test_images = random.sample(devgtest_images, int(num_devtest_datapoints / 2))
testset = set(test_images)

dev_images = [i for i in devtest_images if i not in testset]

devset = set(dev_images)
devtestset= set(devtest_images)

##
# write output

out = { "test" : sorted(test_images),
        "dev" : sorted(dev_images),
        "train" : sorted([i for i in vgitems["images"] if i not in devtestset]) }

split_zipfilename, split_filename = vgpath_obj.vg_traintest_zip_and_filename(write = True)
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
dev_objects = Counter()

train_attrib = Counter()
test_attrib = Counter()
dev_attrib = Counter()

train_rel = Counter()
test_rel = Counter()
dev_rel = Counter()

vgobj = vgiterator.VGIterator(vgcounts = vgitems)

# object counts
for img, frequent_objects in vgobj.each_image_objects():
    for label in frequent_objects:
        if img in trainset: train_objects[label] += 1
        elif img in testset: test_objects[label] += 1
        elif img in devset: dev_objects[label] += 1

# attribute counts
for img, frequent_attrib in vgobj.each_image_attributes():
    for label in frequent_attrib:
        if img in trainset: train_attrib[label] += 1
        elif img in testset: test_attrib[label] += 1
        elif img in devset: dev_attrib[label] += 1

# relation counts
for img, frequent_rel in vgobj.each_image_relations():
    for label in frequent_rel:
        if img in trainset: train_rel[label] += 1
        elif img in testset: test_rel[label] += 1
        elif img in devset: dev_rel[label] += 1

print("--")
print("Number of object instances: Training:", train_objects.total())
print("Number of object instances: Dev:", dev_objects.total())
print("Number of object instances: Test:", test_objects.total())

print("--")
print("Number of attribute instances: Training:", train_attrib.total())
print("Number of attribute instances: Dev:", dev_attrib.total())
print("Number of attribute instances: Test", test_attrib.total())

print("--")
print("Number of relation instances: Training:", train_rel.total())
print("Number of relation instances: Dev:", dev_rel.total())
print("Number of relation instances: Test", test_rel.total())
