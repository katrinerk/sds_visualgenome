# Katrin Erk January 2023
# map between words for Visual Genome objects, attributes and relations
# and indices to be used within the factor graph

class VgitemIndex:
    def __init__(self, object_attr_rel_lists, additional_dict = None):
        self.object_attr_rel_lists = object_attr_rel_lists
        self.num_objects = len(object_attr_rel_lists["objects"])
        self.firstattr = len(object_attr_rel_lists["objects"])
        self.firstrel = self.firstattr + len(object_attr_rel_lists["attributes"])
        self.lastix = len(object_attr_rel_lists["objects"]) + len(object_attr_rel_lists["attributes"]) + len(object_attr_rel_lists["relations"]) - 1
        
        self.objectix = dict((c, i) for i, c in enumerate(object_attr_rel_lists["objects"]))
        self.attrix = dict((c, self.firstattr + i) for i, c in enumerate(object_attr_rel_lists["attributes"]))
        self.relix = dict((c, self.firstrel + i) for i, c in enumerate(object_attr_rel_lists["relations"]))

        # dictionary: index -> (label, type)
        self.additional_dict = additional_dict


    def o2ix(self, o):
        return self.objectix.get(o, None)

    def a2ix(self, a):
        return self.attrix.get(a, None)

    def r2ix(self, r):
        return self.relix.get(r, None)

    def ix2l(self, ix):
        if ix > self.lastix:
            # not an object, attribute, or relation: cloze word?
            if self.additional_dict is not None and ix in self.additional_dict:
                return self.additional_dict[ix]
            else:
                return (None, None)
            
        elif ix >= self.firstrel:
            # relation
            return (self.object_attr_rel_lists["relations"][ix-self.firstrel], "rel")
        
        elif ix >= self.firstattr:
            # attribute
            return (self.object_attr_rel_lists["attributes"][ix-self.firstattr], "att")
        
        else:
            # object
            return (self.object_attr_rel_lists["objects"][ix], "obj")

   
