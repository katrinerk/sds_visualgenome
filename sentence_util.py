# Katrin Erk Feb 2023
# utilities for inspecting sentences and their SDS analyses



import sys
import os
import json
import zipfile
from collections import defaultdict, Counter
import math
import numpy as np
import random
import statistics

import vgiterator
from vgindex import VgitemIndex
from vgpaths import VGPaths

######
# keep Scenario descriptions, and return scenario description as a string
class TopicInfoUtil:
    def __init__(self, vgpath_obj):
        # read topic characterizations
        gensim_zipfilename, overall_filename, topic_filename, word_filename, topicword_filename = vgpath_obj.gensim_out_zip_and_filenames()
        with zipfile.ZipFile(gensim_zipfilename) as azip:
            with azip.open(topic_filename) as f:
                topic_info = json.load(f)

        # store as mapping topic index -> description
        self.topic = dict((e["topic"], e["descr"]) for e in topic_info)

    def topic_info(self, topic_number):
        if topic_number not in self.topic:
            # not found
            return "[dummy topic]"

        else:
            # return string that describes the topic
            return ", ".join([str(word[3:]) + ":" + str(round(p,3)) for word, p in self.topic[topic_number]])
            
        
        
########
# print sentences
class SentencePrinter:
    def __init__(self, vgindex_obj, file = sys.stdout, with_wordliteral_index = False, 
                show_attrel_unary = False, items_per_line = 4):
        
        self.outf = file
        self.vgindex_obj = vgindex_obj
        self.with_wordliteral_index = with_wordliteral_index
        self.show_attrel_unary = show_attrel_unary
        self.items_per_line = items_per_line

    # write sentence to outf
    # if target index is given: optionally mark the target literal with **...**
    # if scenario_concept_assignment is given: show assigned scenario, concept for each word literal
    def write_sentence(self, sentence, targetindex = None, scenario_concept_assignment = None):

        ###
        # sort through the sentence:
        # store the names associated with each discourse referent,
        # and store whether it is an object or not
        # for roles, store the arguments
        wordliterals = [ ] # list of word literals
        dref_names = defaultdict(list) # keep names for discourse referent
        dref_wordtype = { } # obj or rel or attr
        pred_args = defaultdict(list) # for predicate dref, keep argument drefs
        target_dref = None

        for ell in sentence:
            if ell[0] == "w":
                # unary literal, word for either an object or an attribute or a relation
                _, word_id, dref = ell

                # get word ID, and from there, readable word
                word, wordtype = self.vgindex_obj.ix2l(word_id)
                if wordtype is None:
                    raise Exception('failed to look up word ID' + str(word_id))


                # store word literals, in order: word label, discourse referent
                wordliterals.append( ( word, dref) )
                # store label for this discourse referent
                dref_names[ dref].append(word)
                # store word type for this discourse referent
                dref_wordtype[ dref ] = wordtype


            else:
                # binary literal: role
                _, role, dref_h, dref_d = ell
                pred_args[dref_h].append( (role, dref_d))

        ###
        # for words whose type is "xxx", that is, they come from an extra dictionary
        # of cloze words: determine whether they should be att, rel, obj after all
        pairs = list(dref_wordtype.items())
        for dref, wordtype in pairs:
            if wordtype == "xxx":
                if dref not in pred_args:
                    # there are no roles, so call it an object
                    dref_wordtype[ dref ] = "obj"
                elif len(pred_args[dref]) == 1:
                    # one role: call it an attribute
                    dref_wordtype[ dref ] == "att"
                else:
                    # two roles: relation
                    dref_wordtype[dref] = "rel"
            
        
        ###
        # now write word literals for objects, attributes, relations
        for wtype, wtype_txt in [("obj", "Objects"), ("att", "Attributes"), ("rel", "Relations")]:
            if not self.show_attrel_unary and wtype != "obj":
                continue

            print("--", file = self.outf)
            print(wtype_txt + ":", file = self.outf)
            
            output = [ ]
            for wordindex, word_dref in enumerate(wordliterals):
                word, dref = word_dref

                # only writing literals of one type at a time
                if dref_wordtype[dref] != wtype:
                    continue


                readable = word + "(" + str(dref) + ")"

                if targetindex is not None and wordindex == targetindex:
                    readable = "**" + readable + "**"
                    
                if self.with_wordliteral_index:
                    readable = str(wordindex) + ":" + readable

                if scenario_concept_assignment is not None:
                    readable = readable + ",scen:" + str(scenario_concept_assignment["scenario"][wordindex])
                    readable = readable + ",conc:" + self.vgindex_obj.ix2l(scenario_concept_assignment["concept"][wordindex])[0]


                output.append(readable)

            for startindex in range(0, len(output), self.items_per_line):
                print("\t".join(output[startindex:startindex + self.items_per_line]), file = self.outf)
                
            
        # output predicate=-argument tuples for roles
        output = [ ]
        print("--", file = self.outf)
        print("Roles", file = self.outf)

        for dref in sorted(dref_names.keys()):
            if dref_wordtype[dref] != "obj":
                args = [ ]
                for arg in sorted(pred_args[dref]):
                    _, dref_d = arg
                    args.append(",".join(dref_names[dref_d]) + ":" + str(dref_d))

                readable= ",".join(dref_names[dref]) + ":" + str(dref) + "(" + ",".join(args) + ")"

                output.append(readable)

        for startindex in range(0, len(output), self.items_per_line):
            print("\t".join(output[startindex:startindex + self.items_per_line]), file = self.outf)
