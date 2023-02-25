# Katrin Erk January 2023
# counting objects, attributes, relations in Visual Genome
# to select sufficiently frequent data for the SDS+VG experiments

import sys
import os
import json
import zipfile
import nltk
from collections import defaultdict, Counter
import configparser
from argparse import ArgumentParser

from vgpaths import VGPaths

##
# initialize, read global parameters

# where to write the output
parser = ArgumentParser()
parser.add_argument('--vgdata', help="directory where to write frequent-item information", default = "data/")
args = parser.parse_args()

vgpath_obj = VGPaths(vgdata = args.vgdata)

# cutoffs for frequent objects, attributes, relations
config = configparser.ConfigParser()
config.read("settings.txt")

##
# images to use:
# they need to have at least 2 common objects
# or one common object with a common attribute,
# or 2 common object with a common relation
# (though that last case is included in the first)
good_images = set()

##
# count appearances of object labels (which we'll use as objects)
objects_zipfile, objects_file = vgpath_obj.vg_objects_zip_and_filename()
with zipfile.ZipFile(objects_zipfile) as azip:
    with azip.open(objects_file) as f:
        vg_data = json.load(f)
###
# count how often each object (that is, name) appears in VG

objectcount = Counter()

for img in vg_data:
    for obj in img["objects"]:
        for label in obj["names"]:
            objectcount[ label] += 1

freq_objects = set([s for s in objectcount if objectcount[s] >= int(config["Parameters"]["VGFreqCutoffObj"])])
print("num of frequent objects:", len(freq_objects))
print()

print("most common")

for word, freq in objectcount.most_common(20):
    print(word, ":", freq)
    
print()
print("lower end")

print([s for s in objectcount if objectcount[s] == int(config["Parameters"]["VGFreqCutoffObj"])])
print()

##
# determine object IDs of objects with common labels
# and images with at least 2 common objects
# keep image IDs as strings throughout
image_objid = { }

for img in vg_data:
    imgid = str(img["image_id"])
    
    image_objid[imgid] = [o["object_id"] for o in img["objects"] if any(s in freq_objects for s in o["names"])]
    if len(image_objid[imgid]) > 1:
        good_images.add(imgid)

print("Number of good images:", len(good_images))
print()


##
# count appearances of attributes with objects that have frequent-enough labels

attrcount = Counter()

attributes_zipfile, attributes_file = vgpath_obj.vg_attributes_zip_and_filename()
with zipfile.ZipFile(attributes_zipfile) as azip:
    with azip.open(attributes_file) as f:
        vg_data = json.load(f)


for img in vg_data:
    imgid = str(img["image_id"])
    for entry in img.get("attributes", []):
        if "attributes" in entry and entry["object_id"] in image_objid[ imgid]:
            for attr in entry["attributes"]:
                attrcount[ attr ]  += 1

freqattr = set([a for a in attrcount if attrcount[a] >= int(config["Parameters"]["VGFreqCutoffAtt"])])
print("num of frequent attributes", len(freqattr))
print()

print("most common")


for word, freq in attrcount.most_common(20):
    print(word, ":", freq)
    
print()
print("lower end")

print([s for s in attrcount if attrcount[s] == int(config["Parameters"]["VGFreqCutoffAtt"])])
print()

##
# which images have at least one common object with common attribute?
for img in vg_data:
    imgid = str(img["image_id"])
    num_with_common_attr = sum(1 for a in img.get("attributes", []) \
                               if a["object_id"] in image_objid[imgid] and any(s in freqattr for s in a.get("attributes", [])))
    if num_with_common_attr >= 1:
        good_images.add(imgid)

print("Number of good images:", len(good_images))
print()

##
# count frequent relations

relations_zipfile, relations_file = vgpath_obj.vg_relations_zip_and_filename()
with zipfile.ZipFile(relations_zipfile) as azip:
    with azip.open(relations_file) as f:
        vg_data = json.load(f)

relcount = Counter()


for img in vg_data:
    imgid = str(img["image_id"])
    for entry in img.get("relationships", []):
        if entry["subject"]["object_id"] in image_objid[imgid] and \
          entry["object"]["object_id"] in image_objid[imgid]:
          # yes, one of the labels of the subject is a frequent word and
          # one of the labels of the object is a frequent word
            relcount[ entry["predicate"] ] += 1

freqrel = set([a for a in relcount if relcount[a] >= int(config["Parameters"]["VGFreqCutoffRel"])])
print("num of frequent relations", len(freqrel))
print()

print("most common")


for word, freq in relcount.most_common(20):
    print(word, ":", freq)
    
print()
print("lower end")

print([s for s in relcount if relcount[s] == int(config["Parameters"]["VGFreqCutoffRel"])])
print()


##
# write above-cutoff data to data directory
out = { "objects" : sorted(list(freq_objects)),
        "attributes" : sorted(list(freqattr)),
        "relations" : sorted(list(freqrel)),
        "images" : sorted(list(good_images)),
        "image_objid" : image_objid}

vgcounts_zipfilename, vgcounts_filename = vgpath_obj.vg_counts_zip_and_filename(write=True)
with zipfile.ZipFile(vgcounts_zipfilename, "w", zipfile.ZIP_DEFLATED) as azip:
    azip.writestr(vgcounts_filename, json.dumps(out, indent = 1))


