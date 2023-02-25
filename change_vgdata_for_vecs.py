# Katrin Erk January 2023
# 
# Make changed Visual Genome files
# for objects, attributes, relations:
# keep only those images that are part of the training data

import sys
import os
import json
import zipfile
from collections import defaultdict, Counter
import random
from argparse import ArgumentParser

import vgiterator
from vgpaths import VGPaths

parser = ArgumentParser()
parser.add_argument('output', help="directory to write modified VG output to")
parser.add_argument('--vgdata', help="directory with VG data including frequent items, train/test split, topic model", default = "data/")
args = parser.parse_args()


vgpath_obj = VGPaths(vgdata = args.vgdata)

# read frequent object counts
vgcounts_zipfilename, vgcounts_filename = vgpath_obj.vg_counts_zip_and_filename()
with zipfile.ZipFile(vgcounts_zipfilename) as azip:
    with azip.open(vgcounts_filename) as f:
        vgcounts = json.load(f)

# read training/test split
split_zipfilename, split_filename = vgpath_obj.vg_traintest_zip_and_filename()
with zipfile.ZipFile(split_zipfilename) as azip:
    with azip.open(split_filename) as f:
        traintest_split = json.load(f)
    
trainset = set(traintest_split["train"])

# read filenames to write changed VG data to
vg_newobjzip, vg_newattrzip, vg_newrelzip = vgpath_obj.vg_manipulated_filenames(args.output, write = True)

# read current object file 
objects_zipfile, objects_file = vgpath_obj.vg_objects_zip_and_filename()
new_vg_data = [ ]
num_objects = 0
with zipfile.ZipFile(objects_zipfile) as azip:
    with azip.open(objects_file) as f:
        vg_data = json.load(f)

        for img in vg_data:
            imageid = str(img["image_id"])
            if imageid in trainset:
                newimg = {'image_id' : img["image_id"],
                          'objects' : [ o for o in img.get("objects", []) 
                                        if o["object_id"] in vgcounts["image_objid"][imageid] ] }
                num_objects += len(newimg["objects"])
                new_vg_data.append(newimg)            

print("number of objects retained:", num_objects, "in #sentences", len(new_vg_data))

# write new objects file
with zipfile.ZipFile(vg_newobjzip, "w") as ozip:
    ozip.writestr(objects_file, json.dumps(new_vg_data))

# read current attributes file
attributes_zipfile, attributes_file = vgpath_obj.vg_attributes_zip_and_filename()
new_vg_data = [ ]
num_attr = 0

with zipfile.ZipFile(attributes_zipfile) as azip:
    with azip.open(attributes_file) as f:
        vg_data = json.load(f)

        for img in vg_data:
            imageid = str(img["image_id"])
            if imageid in trainset:
                newattrib = [ ]
                for o in img.get("attributes", []):
                    
                    # is this an attribute of an object we are keeping?
                    if o["object_id"] not in vgcounts["image_objid"][imageid]:
                        continue

                    # are any attributes frequent enough?
                    labels = [ell for ell in o.get("attributes", []) if ell in vgcounts["attributes"]]
                    if len(labels) == 0: continue

                    newo = o.copy()
                    newo["attributes"] = labels
                    newattrib.append(newo)
                    num_attr += 1

                if len(newattrib) > 0:
                    new_vg_data.append( {'image_id' : img["image_id"],
                                        "attributes" : newattrib })

# write new attributes file
print("number of attributes retained:", num_attr, "in #sentences", len(new_vg_data))
with zipfile.ZipFile(vg_newattrzip, "w") as ozip:
    ozip.writestr(attributes_file, json.dumps(new_vg_data))

# read original relations file
relations_zipfile, relations_file = vgpath_obj.vg_relations_zip_and_filename()
new_vg_data = [ ]
num_rel = 0
with zipfile.ZipFile(relations_zipfile) as azip:
    with azip.open(relations_file) as f:
        vg_data = json.load(f)
        
        for img in vg_data:
            imageid = str(img["image_id"])
            if imageid in trainset:
                newrel = [ ]
                for o in img.get("relationships", []):

                    # are both arguments objects that we are keeping?
                    if o["subject"]["object_id"] not in vgcounts["image_objid"][imageid] or o["object"]["object_id"] not in vgcounts["image_objid"][imageid]:
                        continue

                    # is the relation label frequent enough?
                    if o["predicate"] not in vgcounts["relations"]:
                        continue

                    newrel.append(o)
                    num_rel += 1
                    
                if len(newrel) > 0:
                    new_vg_data.append( {'image_id' : img["image_id"],
                                         "relationships" : newrel })

# write new relations file
print("number of relations retained:", num_rel, "in #sentences", len(new_vg_data))

with zipfile.ZipFile(vg_newrelzip, "w") as ozip:
    ozip.writestr(relations_file, json.dumps(new_vg_data))
