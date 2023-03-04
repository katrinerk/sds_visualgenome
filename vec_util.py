# Katrin Erk February 2023
# interface to Aurelie's Word2vec vectors

import numpy as np
import re


# using gensim for computing cosines
from gensim.models import KeyedVectors

class VectorInterface:
    def __init__(self, vgpath_obj):
        vecfilename = vgpath_obj.vg_vecfilename()
        self.object_vec = { }
        self.attrib_vec = { }
        self.predarg0_vec = { }
        self.predarg1_vec = { }

        ###
        # read vectors,
        # vectors for objects will be available in
        # object_vec, a dictionary
        # object label -> vector of floating point numbers
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
                    # object
                    label = label[:-2]
                    label = label.replace("_", " ")
                    self.object_vec[ label ] = np.array([ float(v) for v in pieces[1:]])

                elif "(" in label and ")" in label:
                    # relation, curried: relation plus one argument
                    match = re.match("(.+)\((.+),(.+)\)$", label)
                    if match is None:
                        print("Vec Util: could not decompose label", label)
                        continue

                    predicate = match[1]
                    arg0 = match[2]
                    arg1 = match[3]

                    predicate = predicate.replace("|", " ")
                    if arg0 == "-":
                        # pred/arg1 combination
                        if not arg1.endswith(".n"):
                            print("Vec Util: arg should be a noun but isn't", arg1)
                        arg1 = arg1[:-2]
                        arg1 = arg1.replace("_", " ")
                        self.predarg1_vec[ (predicate, arg1) ] = np.array([ float(v) for v in pieces[1:]])

                    elif arg1 == "-":
                        # pred/arg0 combination
                        if not arg0.endswith(".n"):
                            print("Vec Util: arg should be a noun but isn't", arg0)
                        arg0 = arg0[:-2]
                        arg0 = arg0.replace("_", " ")
                        self.predarg0_vec[ (predicate, arg0) ] = np.array([ float(v) for v in pieces[1:]])

                    else:
                        print("Vec Util: could not parse pred/arg entry, did not find - entry", label)


                else:
                    # attribute
                    label = label.replace("|", " ")
                    self.attrib_vec[ label ] = np.array([ float(v) for v in pieces[1:]])

        # make gensim KeyedVector for objects, attributes, predicates to support similarity to centroid
        self.object_kv = KeyedVectors(len(list(self.object_vec.values())[0]))
        self.object_kv.add_vectors(list(self.object_vec.keys()), list(self.object_vec.values()))
        self.object_kv.fill_norms()
        self.object_kv_index_key = dict( (self.object_kv.get_index(key), key) for key in self.object_vec.keys())

        self.attrib_kv = KeyedVectors(len(list(self.attrib_vec.values())[0]))
        self.attrib_kv.add_vectors(list(self.attrib_vec.keys()), list(self.attrib_vec.values()))
        self.attrib_kv.fill_norms()
        self.attrib_kv_index_key = dict( (self.attrib_kv.get_index(key), key) for key in self.attrib_vec.keys())

        self.predarg0_kv = KeyedVectors(len(list(self.predarg0_vec.values())[0]))
        self.predarg0_kv.add_vectors(list(self.predarg0_vec.keys()), list(self.predarg0_vec.values()))
        self.predarg0_kv.fill_norms()
        self.predarg0_kv_index_key = dict( (self.predarg0_kv.get_index(key), key) for key in self.predarg0_vec.keys())
        

        self.predarg1_kv = KeyedVectors(len(list(self.predarg1_vec.values())[0]))
        self.predarg1_kv.add_vectors(list(self.predarg1_vec.keys()), list(self.predarg1_vec.values()))
        self.predarg1_kv.fill_norms()
        self.predarg1_kv_index_key = dict( (self.predarg1_kv.get_index(key), key) for key in self.predarg1_vec.keys())
        
        

    def all_objects_sim_centroid(self, object_labels):
        centroid = self.object_kv.get_mean_vector(object_labels)
        if sum(centroid) == 0.0:
            return None

        # compute similarity from centroid to all objects
        sims = self.object_kv.cosine_similarities(centroid, self.object_kv.get_normed_vectors())

        # and return as list of pairs (object label, sim)
        return [ (self.object_kv_index_key[ix], sim) for ix, sim in enumerate(sims.tolist())]

    # all similarities, to this label, ranked highest to lowest,
    # where type = object, attr, predarg0, predarg1
    def ranked_sims(self, label, elltype):
        if elltype == "object":
            sims = self.object_kv.similar_by_key(label, topn = None)
            return [ (self.object_kv_index_key[ix], sim) for ix, sim in enumerate(sims.tolist())]
        
        elif elltype == "attr":
            sims = self.attrib_kv.similar_by_key(label, topn = None)
            return [ (self.attrib_kv_index_key[ix], sim) for ix, sim in enumerate(sims.tolist())]
        
        elif elltype == "predarg0":
            sims = self.predarg0_kv.similar_by_key(label, topn = None)
            return [ (self.predarg0_kv_index_key[ix], sim) for ix, sim in enumerate(sims.tolist())]
        
        elif elltype == "predarg1":
            sims = self.predarg1_kv.similar_by_key(label, topn = None)
            return [ (self.predarg1_kv_index_key[ix], sim) for ix, sim in enumerate(sims.tolist())]
            
        else:
            raise Exception("unknown label type " + elltype)

                        
            
