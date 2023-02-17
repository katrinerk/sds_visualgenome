# Katrin Erk January 2023
# iterate over images in the Visual Genome,
# either over all its objects, or all its attributes, or all its relations

import sys
import os
import json
import zipfile
from collections import defaultdict, Counter
from vgpaths import VGPaths


class VGIterator:
    # initialize with:
    # - directory where the Visual Genome is located (assumes subdirectories)
    #   v1_2, v1_4),
    # - file containing output from vgcounts, a json listing sufficiently frequent
    #   objects, attributes, relations
    def __init__(self, vgcounts = None):
        self.vgpath_obj = VGPaths()

        if vgcounts is None:
            vgcounts_zipfilename, vgcounts_filename = self.vgpath_obj.vg_counts_zip_and_filename()
            with zipfile.ZipFile(vgcounts_zipfilename) as azip:
                with azip.open(vgcounts_filename) as f:
                    vgcounts = json.load(f)
        
        self.frequent_objects = set(vgcounts["objects"])
        self.frequent_attrib = set(vgcounts["attributes"])
        self.frequent_rel = set(vgcounts["relations"])
        self.image_objid = vgcounts["image_objid"]

    # iterate over each image, yield its sufficiently frequent objects as their names (not synsets)
    # if an object has multiple names, each name is listed seperately.
    #
    # if img_ids is given: yield only images with ids in that set
    def each_image_objects(self, img_ids = None):

        objects_zipfile, objects_file = self.vgpath_obj.vg_objects_zip_and_filename()
        with zipfile.ZipFile(objects_zipfile) as azip:
            with azip.open(objects_file) as f:
                vg_data = json.load(f)
        

        for img in vg_data:
            
            imageid = str(img["image_id"])
            
            if img_ids is not None and imageid not in img_ids: continue
                
            # image ID, list of object labels
            yield (imageid,
                   [l for o in img.get("objects", []) for l in o.get("names", []) \
                   if o["object_id"] in self.image_objid[imageid] and l in self.frequent_objects])

    # iterate over each image, yield its sufficiently frequent attributes as their names (not synsets)
    # if an attribute has multiple names, each name is listed seperately.
    #
    # if img_ids is given: yield only images with ids in that set
    def each_image_attributes(self, img_ids = None):

        attributes_zipfile, attributes_file = self.vgpath_obj.vg_attributes_zip_and_filename()
        with zipfile.ZipFile(attributes_zipfile) as azip:
            with azip.open(attributes_file) as f:
                vg_data = json.load(f)

        for img in vg_data:
            imageid = str(img["image_id"])
            
            if img_ids is not None and imageid not in img_ids: continue

            # image ID, list of attribute labels
            yield (imageid,
                  [l for o in img.get("attributes", []) for l in o.get("attributes", []) \
                  if l in self.frequent_attrib and o["object_id"] in self.image_objid[imageid]])
        
        
    # iterate over each image, yield its sufficiently frequent relations as their names (not synsets)
    # if a relation has multiple names, each name is listed seperately.
    #
    # if img_ids is given: yield only images with ids in that set
    def each_image_relations(self, img_ids = None):

        relations_zipfile, relations_file = self.vgpath_obj.vg_relations_zip_and_filename()
        with zipfile.ZipFile(relations_zipfile) as azip:
            with azip.open(relations_file) as f:
                vg_data = json.load(f)
        
        for img in vg_data:
            imageid = str(img["image_id"])
            
            if img_ids is not None and imageid not in img_ids: continue
                
            yield (imageid,
                    [o["predicate"] for o in img.get("relationships", []) \
                    if o["predicate"] in self.frequent_rel and \
                    o["subject"]["object_id"] in self.image_objid[imageid] and \
                    o["object"]["object_id"] in self.image_objid[imageid] ] )

    # iterate over each image, yield its sufficiently frequent objects as pairs (object ID, object names)
    # if an object has multiple names, they are all listed together 
    def each_image_objects_full(self, img_ids = None):

        objects_zipfile, objects_file = self.vgpath_obj.vg_objects_zip_and_filename()
        with zipfile.ZipFile(objects_zipfile) as azip:
            with azip.open(objects_file) as f:
                vg_data = json.load(f)
        

        for img in vg_data:
            
            imageid = str(img["image_id"])
            
            if img_ids is not None and imageid not in img_ids: continue
                
            # image ID, list of object labels
            objects = []
            for o in img.get("objects", []):
                
                # object not one we keep track of?
                if o["object_id"] not in self.image_objid[imageid]: continue

                # get frequent names of this object -- there should be some
                names = [l for l in o.get("names") if l in self.frequent_objects]
                if len(names) == 0: print("object iterator: Warning: unexpectedly no valid names found for", o["object_id"])
                else: objects.append( (o["object_id"], names))

            # done with this image
            if len(objects) > 0: yield (imageid, objects)
        

    # iterate over each image, yield its sufficiently frequent attributes with arguments:
    # lists (attribute names, object ID) 
    def each_image_attributes_full(self, img_ids = None):

        attributes_zipfile, attributes_file = self.vgpath_obj.vg_attributes_zip_and_filename()
        with zipfile.ZipFile(attributes_zipfile) as azip:
            with azip.open(attributes_file) as f:
                vg_data = json.load(f)

        for img in vg_data:
            imageid = str(img["image_id"])

            if img_ids is not None and imageid not in img_ids: continue
                
            attr = [ ]
            for o in img.get("attributes", []):
                
                # object not one we keep track of?
                if o["object_id"] not in self.image_objid[imageid]: continue
                    
                # can we get any names for the attribute that are above threshold?
                prednames = [l for l in o.get("attributes", []) if l in self.frequent_attrib]
                if len(prednames) > 0:
                    attr.append( (prednames, o["object_id"]))
                                  
            if len(attr) > 0: yield(imageid, attr)

    # iterate over each image, yield its sufficiently frequent relations with arguments:
    # lists (relation names, subject ID, object IDs) 
    def each_image_relations_full(self, img_ids = None):

        relations_zipfile, relations_file = self.vgpath_obj.vg_relations_zip_and_filename()
        with zipfile.ZipFile(relations_zipfile) as azip:
            with azip.open(relations_file) as f:
                vg_data = json.load(f)


        for img in vg_data:
            imageid = str(img["image_id"])

            if img_ids is not None and imageid not in img_ids: continue
                
            rel = [ ]
            for o in img.get("relationships", []):
                
                # subject or object not one we keep track of?
                if o["subject"]["object_id"] not in self.image_objid[imageid] or o["object"]["object_id"] not in self.image_objid[imageid]:
                    continue
                
                # is the relation name above threshold?
                if o["predicate"] in self.frequent_rel:
                    rel.append( (o["predicate"], o["subject"]["object_id"], o["object"]["object_id"]) )
                                  
            if len(rel) > 0: yield(imageid, rel)
        
