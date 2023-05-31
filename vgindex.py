# Katrin Erk January 2023
# map between words for Visual Genome objects, attributes and relations
# and indices to be used within the factor graph

from vgnames import VGOBJECTS, VGATTRIBUTES, VGRELATIONS 


class VgitemIndex:
    # initialization with:
    # dictionary with keys "objects", "attributes", "relations" keeping lists of frequent labels
    # additional_dict: set to handle cloze words.
    # this is a dictionary word index -> label, type
    def __init__(self, object_attr_rel_lists, additional_dict = None):
        self.object_attr_rel_lists = object_attr_rel_lists
        self.num_objects = len(object_attr_rel_lists[VGOBJECTS])
        self.firstattr = len(object_attr_rel_lists[VGOBJECTS])
        self.firstrel = self.firstattr + len(object_attr_rel_lists[VGATTRIBUTES])
        self.lastix = len(object_attr_rel_lists[VGOBJECTS]) + len(object_attr_rel_lists[VGATTRIBUTES]) + len(object_attr_rel_lists[VGRELATIONS]) - 1
        
        self.objectix = dict((c, i) for i, c in enumerate(object_attr_rel_lists[VGOBJECTS]))
        self.attrix = dict((c, self.firstattr + i) for i, c in enumerate(object_attr_rel_lists[VGATTRIBUTES]))
        self.relix = dict((c, self.firstrel + i) for i, c in enumerate(object_attr_rel_lists[VGRELATIONS]))

        # dictionary: index -> (label, type)
        self.additional_dict = additional_dict


    ##
    # map object label or attribute label or relation label to index
    def o2ix(self, o):
        return self.objectix.get(o, None)

    def a2ix(self, a):
        return self.attrix.get(a, None)

    def r2ix(self, r):
        return self.relix.get(r, None)

    ##
    # test if a label is an object label, attribute label, or relation label
    def isobj(self, o):
        return o in self.objectix or (self.additional_dict is not None and (o, VGOBJECTS) in self.additional_dict)

    def isatt(self, a):
        return a in self.attrix or (self.additional_dict is not None and (a, VGATTRIBUTES) in self.additional_dict)

    def isrel(self, r):
        return r in self.relix or (self.additional_dict is not None and (r, VGRELATIONS) in self.additional_dict)

    ##
    # map an index to a pair (label, type)
    # where type is one of "obj", "att", "rel"
    def ix2l(self, ix):
        if ix > self.lastix:
            # not an object, attribute, or relation: cloze word?
            if self.additional_dict is not None and ix in self.additional_dict:
                return self.additional_dict[ix]
            else:
                return (None, None)
            
        elif ix >= self.firstrel:
            # relation
            return (self.object_attr_rel_lists[VGRELATIONS][ix-self.firstrel], VGRELATIONS)
        
        elif ix >= self.firstattr:
            # attribute
            return (self.object_attr_rel_lists[VGATTRIBUTES][ix-self.firstattr], VGATTRIBUTES)
        
        else:
            # object
            return (self.object_attr_rel_lists[VGOBJECTS][ix], VGOBJECTS)

   
