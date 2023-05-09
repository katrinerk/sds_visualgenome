# Katrin Erk January 2023:
# helper functions for different scripts that 
# make specification for an SDS system
# using Visual Genome data:
# scenario/concept probabilities from an LDA topic model,
# selectional preferences from co-occurrence counts of
# attributes with objects, and relations with object pairs,
# training/test sentences

import sys
import os
import json
import zipfile
from collections import defaultdict, Counter
import nltk
import math
import numpy as np
import random

import vgiterator
from vgindex import VgitemIndex
from vec_util import VectorInterface

#############
# class for reading data for SDS parameters
class VGParam:
    # read initial data
    # top scenarios per concept: only this many scenarios per concept will be listed
    # frequentobj: counts of frequent objects, attributes, relations, will be read from file if not given
    # selpref_vectors: if True, compute selectional preferences using vectors rather than observed
    #   relative frequencies
    def __init__(self, vgpath_obj, frequentobj = None, selpref_vectors = True):
        # object with paths to VG data
        self.vgpath_obj = vgpath_obj

        if frequentobj is None:

            vgcounts_zipfilename, vgcounts_filename = self.vgpath_obj.vg_counts_zip_and_filename()
            with zipfile.ZipFile(vgcounts_zipfilename) as azip:
                with azip.open(vgcounts_filename) as f:
                    self.vgobjects_attr_rel = json.load(f)
        else:
            self.vgobjects_attr_rel = frequentobj

        self.vgobj = vgiterator.VGIterator(self.vgobjects_attr_rel)
        self.vgindex = VgitemIndex(self.vgobjects_attr_rel)
        
        split_zipfilename, split_filename = self.vgpath_obj.vg_traintest_zip_and_filename()
        with zipfile.ZipFile(split_zipfilename) as azip:
            with azip.open(split_filename) as f:
                traintest_split = json.load(f)
    
        self.trainset = set(traintest_split["train"])
        self.testset = set(traintest_split["test"])

        # read vectors?
        self.selpref_vectors = selpref_vectors
        self.vec_obj = VectorInterface(self.vgpath_obj) if self.selpref_vectors else None

    # obtain data and return it:
    # * global parameters
    # * concept/scenario log probabilities, from topic modeling output
    # * word/concept log probabilities, empty
    # * selectional preferences, from predicate/argument co-occurrence counts
    def get(self):
        # list of : global parameters, concept/scenario log prob, word/concept log prob, selectional preferences
        retv = [ ]
        
        #########
        # read topic modeling output

        gensim_zipfilename, overall_filename, topic_filename, word_filename, topicword_filename = self.vgpath_obj.gensim_out_zip_and_filenames()
        with zipfile.ZipFile(gensim_zipfilename) as azip:
            # Dirichlet alpha values, number of topics
            with azip.open(overall_filename) as f:
                topic_param = json.load(f)

            # ordered list of words as they appear in topics. need to be mapped to concepts and concept indices
            with azip.open(word_filename) as f:
                topic_wordlist = json.load(f)
        
            # term-document matrix, one row per topic, one column per word
            # these are log probabilities
            with azip.open(topicword_filename) as f:
                term_topic_matrix = np.array(json.load(f))

        num_concepts = len(self.vgobjects_attr_rel["objects"]) + len(self.vgobjects_attr_rel["attributes"]) + len(self.vgobjects_attr_rel["relations"])
        
        ##
        # global data: alpha, number of concepts (objects, attributes, relations), number of scenarios, number of words
        # add one topic: for words that have no topic assigned
        num_scenarios = topic_param["num_topics"] + 1
        retv.append({"num_concepts" : num_concepts,
                      "num_scenarios" : num_scenarios,
                      "num_words" : num_concepts,
                      "dirichlet_alpha" : topic_param["alpha"]})

        ##
        # scenario-concept file

        # map each word from the topic model to a concept index
        # words have the form "obj"word or "att" word or "rel" word

        # list of concept indices in the order of columns in the term-topic matrix
        conceptindices_tm = []
        # map gensim's word indices to our word indices
        for word in topic_wordlist:
            if word[:3] =="obj":
                ix = self.vgindex.o2ix(word[3:])
                if ix is None:
                    raise Exception("lookup error for topic word, object", word[3:])
                conceptindices_tm.append(ix)
        
            elif word[:3] == "att":
                ix = self.vgindex.a2ix(word[3:])
                if ix is None:
                    raise Exception("lookup error for topic word, attribute", word[3:])
                conceptindices_tm.append(ix)
        
            elif word[:3] == "rel":
                ix = self.vgindex.r2ix(word[3:])
                if ix is None:
                    raise Exception("lookup error for topic word, relation", word[3:])
                conceptindices_tm.append(ix)

            else:
                raise Exception("lookup error, unknown word type", word)

        # for each concept index, store:
        # "scenarios" : list of appropriate scenarios
        # "weights" : log probabilities logP(concept|scenario)
        concept_scenario = {}
        for columnindex, conceptindex in enumerate(conceptindices_tm):
            topic_weights = term_topic_matrix[:, columnindex]
            # gensim wrapper stores log probabilities, so the weights are already log probabilities
            concept_scenario[ conceptindex ] =  {"scenario" : list(range(num_scenarios - 1)),
                                                 "weight" : [ w.item() for w in topic_weights ] }

        # sanity check: do we have scenarios for all frequent concepts?
        # if not, assign it the very last scenario, with probability 1
        for i in range(self.vgindex.lastix + 1):
            if i not in concept_scenario:
                concept_scenario[ i ] = {"scenario" : [ num_scenarios - 1 ],
                                         "weight" : [ 0.0 ]}


        # done with scenario-concept probs
        retv.append( concept_scenario)

        ##########3
        # word/concept log probabilities:
        # none stored because in the base case,
        # every word just corresponds to a concept with the same name
        # cloze word IDs -> concept ID -> logprob
        retv.append( { })
        
        
        ##########3
        # selpref
    
        attr_count = nltk.ConditionalFreqDist()
        rel_count_subj = nltk.ConditionalFreqDist()
        rel_count_obj = nltk.ConditionalFreqDist()

        # collect labels of potential arguments
        img_objid_label = { }

        for imgid, objects in self.vgobj.each_image_objects_full():
            img_objid_label[imgid] = { }
            for objid, names in objects:
                img_objid_label[imgid][objid] = names

        # now iterate through attributes to collect their arguments
        for imgid, attr_arg in self.vgobj.each_image_attributes_full(img_ids = self.trainset):
            for attrnames, objid in attr_arg:
                if imgid not in img_objid_label or objid not in img_objid_label[imgid]:
                    print("Warning: unexpectedly didn't find attr argument", imgid, objid)
                    continue
        
                for a in attrnames: 
                    for l in img_objid_label[imgid][objid]:
                        attr_count[a][l] += 1

        # and same for relations
        for imgid, rel_arg in self.vgobj.each_image_relations_full(img_ids = self.trainset):
            for predname, subjid, objid in rel_arg:
                if imgid not in img_objid_label or objid not in img_objid_label[imgid] or subjid not in img_objid_label[imgid]:
                    print("Warning: unexpectedly didn't find rel argument", imgid, subjid, objid)
                    continue
        
                for l in img_objid_label[imgid][subjid]:
                    rel_count_subj[predname][l] += 1
                for l in img_objid_label[imgid][objid]:
                    rel_count_obj[predname][l] += 1

        if self.selpref_vectors:
            selpref_json = self.compute_selpref_vectors(attr_count, rel_count_subj, rel_count_obj)
        else:
            selpref_json = self.compute_selpref_relfreq(attr_count, rel_count_subj, rel_count_obj)

        retv.append(selpref_json)

        return retv

    ####
    # compute selectional preferences as relative frequency, return as dictionary
    # arg0/arg1 -> dictionary with entries "config": list of arguments, "weight": list of matching weights
    def compute_selpref_relfreq(self, attr_count, rel_count_subj, rel_count_obj):
        selpref_json = { "arg0" : {"config" : [], "weight" : []},
                         "arg1" :  {"config" : [], "weight" : [] } }

        # store attribute/argument pairs and their weights,
        # rel/subj pairs, rel/obj pairs
        for cfd, predtype, label in [ (attr_count, "attr", "arg1"), (rel_count_subj, "rel", "arg0"), (rel_count_obj, "rel", "arg1")]:
            for pred in cfd.conditions():
                if predtype == "attr":
                    predix = self.vgindex.a2ix(pred)
                    if predix is None:
                        print("no index found for attrib", pred)
                else:
                    predix= self.vgindex.r2ix(pred)
                    if predix is None:
                        print("no index found for rel", pred)
            
                normalizer = math.log(sum(cfd[pred].values()))
                for arg, count in cfd[pred].items():
                    selpref_json[label]["config"].append( (predix, self.vgindex.o2ix(arg)) )
                    selpref_json[label]["weight"].append( math.log(count) - normalizer)

        return selpref_json

    ####
    # compute selectional preferences as similarity to centroid vector, return as dictionary
    # arg0/arg1 -> dictionary with entries "config": list of arguments, "weight": list of matching weights
    def compute_selpref_vectors(self, attr_count, rel_count_subj, rel_count_obj):
        selpref_json = { "arg0" : {"config" : [], "weight" : []},
                         "arg1" :  {"config" : [], "weight" : [] } }

        # store attribute/argument pairs and their weights,
        # rel/subj pairs, rel/obj pairs
        for cfd, predtype, label in [ (attr_count, "attr", "arg1"), (rel_count_subj, "rel", "arg0"), (rel_count_obj, "rel", "arg1")]:
            for pred in cfd.conditions():
                # predicate index for this predicate
                if predtype == "attr":
                    predix = self.vgindex.a2ix(pred)
                    if predix is None:
                        print("no index found for attrib", pred)
                else:
                    predix= self.vgindex.r2ix(pred)
                    if predix is None:
                        print("no index found for rel", pred)

                # compute centroid
                # all words
                # wordlist = list(cfd[pred].keys())
                first_n = min([30, len(list(cfd[pred].keys()))])
                wordlist = [w for w, c in cfd[pred].most_common(first_n)]
                # print("HIER", first_n, len(wordlist))
                objectlabels_and_sims = self.vec_obj.all_objects_sim_centroid(wordlist)

                # centroid was zero, nothing to be done for this predicate
                if objectlabels_and_sims is None:
                    continue

                for arg, sim in objectlabels_and_sims:
                    if sim > 0:
                        selpref_json[label]["config"].append( (predix, self.vgindex.o2ix(arg)))
                        selpref_json[label]["weight"].append( sim)
                
        return selpref_json
                    

    #####
    # write parameters to files
    def write(self, global_param, concept_scenario, word_concept, selpref_json):

        ##
        # global file: alpha, number of concepts (objects, attributes, relations), number of scenarios, number of words
        filenames = self.vgpath_obj.sds_filenames(write = True)
        with open(filenames["general"], "w") as outf:
            print(json.dumps(global_param), file = outf)

        ##
        # scenario-concept file

        scenario_zipfile, scenario_file = filenames["scenario_concept"]
        with zipfile.ZipFile(scenario_zipfile, "w", zipfile.ZIP_DEFLATED) as azip:
            azip.writestr(scenario_file, json.dumps(concept_scenario))

        ##
        # word file
        word_zipfile, word_file = filenames["word_concept"]
        with zipfile.ZipFile(word_zipfile, "w", zipfile.ZIP_DEFLATED) as azip:
            azip.writestr(word_file, json.dumps(word_concept))

        
        ##########3
        # selpref file

        selpref_zipfile, selpref_file = filenames["selpref"]     
        with zipfile.ZipFile(selpref_zipfile, "w", zipfile.ZIP_DEFLATED) as azip:
            azip.writestr(selpref_file, json.dumps(selpref_json))

##############################
class VGSentences:
    def __init__(self, vgpath_obj):
        self.vgpath_obj = vgpath_obj

    def each_testsentence(self, vgobj, vgobjects_attr_rel = None, traintest_split = None):

        if vgobjects_attr_rel is None:
            vgcounts_zipfilename, vgcounts_filename = self.vgpath_obj.vg_counts_zip_and_filename()
            with zipfile.ZipFile(vgcounts_zipfilename) as azip:
                with azip.open(vgcounts_filename) as f:
                    vgobjects_attr_rel = json.load(f)

        if traintest_split is None:
            split_zipfilename, split_filename = vgpath_obj.vg_traintest_zip_and_filename()
            with zipfile.ZipFile(split_zipfilename) as azip:
                with azip.open(split_filename) as f:
                    traintest_split = json.load(f)
    
        testset = set(traintest_split["test"])
        vgindex = VgitemIndex(vgobjects_attr_rel)

        ##########
        image_objects = { }
        image_attr = { }
        image_rel = { }

        for img, obj in vgobj.each_image_objects_full(img_ids = testset):
            image_objects[img] = [ (ell, objid) for objid, names in obj for ell in names]

        for img, attr in vgobj.each_image_attributes_full(img_ids = testset):
            image_attr[img] = [ (ell, objid) for names, objid in attr for ell in names]

        for img, rel in vgobj.each_image_relations_full(img_ids = testset):
            image_rel[img] = rel

        # sanity checks
        found = 0
        notfound = 0
        for img in image_objects:
            this_image_notfound = [ ]
            # all object IDs for this image
            ids = [id for _, id in image_objects[img]]

            for attr, objid in image_attr.get(img, []):
                if objid not in ids:
                        notfound += 1
                        this_image_notfound.append( ("attr", attr, objid))
                else: found += 1
            for rel, subjid, objid in image_rel.get(img, []):
                if subjid not in ids:
                        notfound += 1
                        this_image_notfound.append( ("relsubj", rel, subjid))
                else: found += 1
                if objid not in ids:
                        notfound += 1
                        this_image_notfound.append( ("relobj", rel, objjid))
                else: found += 1


        if notfound > 0:
            print("SDS input util warning: some argument IDs not matching any object IDs", notfound, "out of", found + notfound)

        ######
        # transform to final format,
        # yield one sentence at a time

        num_empty = 0
        num_nonempty = 0

        for img, objects in image_objects.items():

            # sentence: list of lists [objlabelindex, discourseref]
            # or [attributelabelindex, discourseref] or [relationlabelindex, discourseref]
            # or [rolelabelindex, discourseref, discourseref]
            words = [ ]
            roles = [ ]

            # add objects
            objectid_discourseref = { }
            for discourseref, objectlabel_id in enumerate(objects):
                objectlabel, object_id = objectlabel_id

                # store discourse referent for this object ID
                objectid_discourseref[ object_id] = discourseref

                # determine label index
                objectlabelindex = vgindex.o2ix(objectlabel)
                if objectlabelindex is None:
                    raise Exception("unknown object " + str(objectlabel))

                # unary condition: label of object
                words.append( ["w", objectlabelindex, discourseref] )

            # next discourse referent: len(objects
            discourseref_offset = len(objects)

            # add attributes
            for almost_discourseref, attr_with_arg in enumerate(image_attr.get(img, [])):
                attr_discourseref = almost_discourseref + discourseref_offset
                attrlabel, object_id = attr_with_arg

                # detemine label index
                attrlabelindex = vgindex.a2ix(attrlabel)
                if attrlabelindex is None:
                    raise Exception("unknown attribute " + str(attrlabel))

                # determine discourse referent for argument
                if object_id not in objectid_discourseref:
                    # discrepancy between attribute listings and object listings.
                    # skip this attribute
                    continue

                arg_dref = objectid_discourseref[object_id]

                # unary condition: label of attribute
                words.append( ["w", attrlabelindex, attr_discourseref])
                # binary condition: object ID is argument of role of attribute
                roles.append( ["r", "arg1", attr_discourseref, arg_dref] )

            # add relations
            discourseref_offset = discourseref_offset + len(image_attr.get(img, []))


            for almost_discourseref, rel_with_arg in enumerate(image_rel.get(img, [])):
                rel_discourseref = almost_discourseref + discourseref_offset
                rellabel, arg0_id, arg1_id = rel_with_arg

                # determine label index
                rellabelindex = vgindex.r2ix(rellabel)
                if rellabelindex is None:
                    raise Exception("unknown relation " + str(rellabel))

                # determine discourse referents for arguments
                if arg0_id not in objectid_discourseref:
                    # discrepancy between relations file and attributes file:
                    # skip this relation
                    continue

                arg0_dref = objectid_discourseref[arg0_id]

                if arg1_id not in objectid_discourseref:
                    # discrepancy between relations file and attributes file:
                    # skip this relation
                    continue

                arg1_dref = objectid_discourseref[arg1_id]

                # unary condition: label of relation
                words.append( ["w", rellabelindex, rel_discourseref])
                # binary conditions: arguments of the relation
                roles.append( ["r", "arg0", rel_discourseref, arg0_dref] )
                roles.append( ["r", "arg1", rel_discourseref, arg1_dref] )

            if len(words) == 0:
                num_empty += 1
            else:
                yield(img, words, roles)
                num_nonempty += 1

        if num_empty > 0:
            print("SDS input util warning: empty sentences found", num_empty, "out of", num_empty + num_nonempty)


    # set cap to None to no capping
    def write(self, sentences, sentlength_cap = 25):

        # sentence literals:
        # object labels:
        # ("w", object label index, discourse referent)
        #
        # attributes:
        # ("w", attribute label index, attribute discourse referent)
        # ("r", "arg1", attribute discourse referent, argument discourse referent)
        #
        # relations:
        # ("w", relation label index, relation discourse referent)
        # ("r", "arg0", relation discourse referent, argument 0 discourse referent)
        # ("r", "arg1", relation discourse referent, argument 1 discourse referent)

        outjson = []
        for sentid, words, words_to_keep, roles in sentences:
            # print("sentlen", len(words) + len(words_to_keep), "keeping:", len(words_to_keep), "shortening:", len(words))

            # cut sentence down to maximum length
            if sentlength_cap is not None and len(words) + len(words_to_keep) > sentlength_cap and len(words) > 0:
                if len(words_to_keep) >= sentlength_cap:
                    words, roles = self.remove_from_sent(words, roles, additional_words = words_to_keep, literals = words)
                    words = words_to_keep + words
                    
                else:
                    # randomly remove words.
                    ##
                    # start with words that aren't involved in any
                    # attributes or relations

                    # discourse referents that are heads or dependents
                    # in a role relation
                    predarg_drefs = set()
                    for _, _, hdref, adref in roles:
                        predarg_drefs.add(hdref)
                        predarg_drefs.add(adref)

                    non_predarg_literals = [ (w, oid, dref) for w, oid, dref in words if dref not in predarg_drefs ]
                    num_to_remove = min(len(non_predarg_literals), len(words_to_keep) + len(words) - sentlength_cap)
                    literals_to_remove = random.sample(non_predarg_literals, num_to_remove)
                    words, roles = self.remove_from_sent(words, roles, additional_words = words_to_keep, literals = literals_to_remove)
                    
                    ##
                    # still too many words? then remove arbitrary words at random,
                    # one by one, because now the removal of one word can trigger
                    # the removal of syntactically adjacent words
                    # remove words, one at a time.
                    while len(words) + len(words_to_keep)  > sentlength_cap and len(words) > 0:
                        # sample a random literal to remove
                        literal_to_remove = random.choice(words)
                        # print("removing", literal_to_remove)

                        # do the removal
                        words, roles = self.remove_from_sent(words, roles, additional_words = words_to_keep, literals = [ literal_to_remove])
                        
                    # print("after shortening", len(words), len(words) + len(words_to_keep))
                    words = words_to_keep + words

            else:
                words = words_to_keep + words

            # drefs_unary = set(d for _, _, d in words + words_to_keep)
            # if any(d not in drefs_unary for _, _, _, d in roles) or any(d not in drefs_unary for _, _, d, _ in roles):
            #     print("discourse referent without unary constraint", sentid)
            #     # HIER
            #     for _, _, d1, d2 in roles:
            #         if d1 not in drefs_unary: print("missing unary for", d1)
            #         if d2 not in drefs_unary: print("missing unary for", d2)
            #     print("------words to keep")
            #     for w in words_to_keep: print(w)
            #     print("-------other words")
            #     for w in words: print(w)
            #     print("---------roles")
            #     for r in roles: print(r)
            #     sys.exit(0)
            
            outjson.append( { "sentence_id" : sentid, "sentence" : words  + roles } )
        
        zipfilename, filename = self.vgpath_obj.sds_sentence_zipfilename(write = True)
        with zipfile.ZipFile(zipfilename, "w", zipfile.ZIP_DEFLATED) as azip:
            azip.writestr(filename, json.dumps(outjson))

    def remove_from_sent(self, words, roles, oids = None, literals = None, additional_words = [ ]):
        if oids is None and literals is None:
            raise Exception("SDS input util error: need either object IDs or literals for removal")

        if oids is not None:
            literals_to_remove = [[w, oid, dref] for w, oid, dref in words if oid in oids]
        else:
            literals_to_remove = literals

        
        # which referents are mentioned among the literals that we aren't removing?
        referents_to_keep = set(lit[2] for lit in words + additional_words if lit not in literals_to_remove )
        # which referents are only mentioned in removed literals, or are predicates
        # that have an argument we're removing
        referents_to_remove = set(dref for _, _, dref in literals_to_remove if dref not in referents_to_keep)
        for _, _, headref, argref in roles:
            if argref in referents_to_remove:
                referents_to_remove.add(headref)


        # what are we keeping? non-removed word literals, except when they mention referents to remove, plus
        # role literals, except when they mention a referent to remove
        words = [ell for ell in words if ell not in literals_to_remove and ell[2] not in referents_to_remove]
        roles = [ell for ell in roles if ell[2] not in referents_to_remove and ell[3] not in referents_to_remove]

        return(words, roles)
                    


        
