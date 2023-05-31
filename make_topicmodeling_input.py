# Katrin Erk January 2023
# make input for topic modeling with Gensim from Visual Genome data
# using an existing train/test split of images,
# write bag-of-words representations of objects, attributes, and relations in each training image
# transformed using a gensim dictionary

import sys
import os
import zipfile
import json
import pickle
from collections import defaultdict, Counter
import nltk
import math
import gensim
from argparse import ArgumentParser

from vgnames import VGOBJECTS, VGATTRIBUTES, VGRELATIONS 
from vgpaths import VGPaths
import vgiterator


parser = ArgumentParser()
parser.add_argument('--vgdata', help="directory with VG data including frequent items, train/test split, topic model", default = "data/")
parser.add_argument('--filter', help="filter extremes in topic modeling input? default: False", action = "store_true")
args = parser.parse_args()


vgpath_obj = VGPaths(vgdata = args.vgdata)


vgcounts_zipfilename, vgcounts_filename = vgpath_obj.vg_counts_zip_and_filename()
with zipfile.ZipFile(vgcounts_zipfilename) as azip:
    with azip.open(vgcounts_filename) as f:
        vgobjects_attr_rel = json.load(f)

split_zipfilename, split_filename = vgpath_obj.vg_traintest_zip_and_filename()
with zipfile.ZipFile(split_zipfilename) as azip:
    with azip.open(split_filename) as f:
        traintest_split = json.load(f)
    
trainset = set(traintest_split["train"])

print("making iterator")
vgobj = vgiterator.VGIterator(vgcounts = vgobjects_attr_rel)

##
# read all training images, collect objects, attributes, relations
# above frequency threshold
img_contents = {}

print("reading training images")
for img, obj in vgobj.each_image_objects(img_ids = trainset):
    img_contents[img] = [VGOBJECTS + o for o in obj]

for img, attr in vgobj.each_image_attributes(img_ids = trainset):
    for a in attr:
        img_contents[img].append(VGATTRIBUTES + a)

for img, rel in vgobj.each_image_relations(img_ids = trainset):
    for r in rel:
        img_contents[img].append(VGRELATIONS + r)


##
# make a Gensim-specific mapping from words to numeric values for our corpus
print("making gensim dictionary")
gensim_dictionary = gensim.corpora.Dictionary(v for v in img_contents.values())
if args.filter:
    print("Filtering the dictionary")
    gensim_dictionary.filter_extremes()

print("writing files")
gensimdict_zipfilename, gensimdict_filename = vgpath_obj.gensim_dict_zip_and_filename( write = True)
with zipfile.ZipFile(gensimdict_zipfilename, "w", zipfile.ZIP_DEFLATED) as azip:
    azip.writestr(gensimdict_filename, pickle.dumps(gensim_dictionary))

##
# write results to file as gensim input for topic modeling
zip_corpusfilename = vgpath_obj.gensim_corpus_zipfilename(write=True)
with zipfile.ZipFile(zip_corpusfilename, "w", zipfile.ZIP_DEFLATED) as azip:
    for img, contents in img_contents.items():
        filename = vgpath_obj.gensim_corpus_filename(img)
        azip.writestr(filename, json.dumps( gensim_dictionary.doc2bow(contents)))
    

