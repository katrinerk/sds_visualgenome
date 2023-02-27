# Katrin Erk February 2023
# interface to Aurelie's Word2vec vectors
import numpy as np
# using gensim for computing cosines
from gensim.models import KeyedVectors

class VectorInterface:
    def __init__(self, vgpath_obj):
        vecfilename = vgpath_obj.vg_vecfilename()
        object_vec = { }

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
                    label = label[:-2]
                    label = label.replace("_", " ")
                    object_vec[ label ] = np.array([ float(v) for v in pieces[1:]])

        # make gensim KeyedVector object
        self.object_kv = KeyedVectors(len(list(object_vec.values())[0]))
        self.object_kv.add_vectors(list(object_vec.keys()), list(object_vec.values()))
        self.object_kv.fill_norms()
        self.object_kv_index_key = dict( (self.object_kv.get_index(key), key) for key in object_vec.keys())
        

    def all_objects_sim_centroid(self, object_labels):
        centroid = self.object_kv.get_mean_vector(object_labels)
        if sum(centroid) == 0.0:
            return None

        # compute similarity from centroid to all objects
        sims = self.object_kv.cosine_similarities(centroid, self.object_kv.get_normed_vectors())

        # and return as list of pairs (object label, sim)
        return [ (self.object_kv_index_key[ix], sim) for ix, sim in enumerate(sims.tolist())]
    
    

