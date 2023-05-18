# Situation description systems, with pgmax:
# multi-sentence discourse
# can have presupposed material, labeled "prew" or "prer" for
# presupposed words or roles
#
#

import sys
import itertools
from argparse import ArgumentParser
import json
import zipfile
from collections import defaultdict, Counter
import math
import numpy as np
import copy
import os
import timeit
import statistics
import configparser

from vgpaths import VGPaths

from dirichlet_multinomial import logddirmult
from my_pgmax import MyFactorGraphWrapper

from sds_core import SDS


    
################################
################################
class SDSD(SDS):
                                                    
      

    ########################3
    # read each sentence from zipped archive,
    # do sanity check
    #
    #
    # format: (sentence_id, sentence)
    #    where a sentence is a list of literals
    #    where a literal is a tuple, one of the following shapes:
    #
    #    ("w", word label index, discourse referent)
    #    ("r", "arg0", pred discourse referent, argument 0 discourse referent)
    #    ("prew", word label index, discourse referent)
    #    ("prer", "arg0", pred discourse referent, argument 0 discourse referent)
    def each_sentence_json(self, verbose = False):
        inputzipfilename, filename = self.vgpath_obj.sds_sentence_zipfilename()
        with zipfile.ZipFile(inputzipfilename) as azip:
            with azip.open(filename) as f:
                sentences = json.load(f)

        malformed_sents = 0
        reflexive_sents = 0

        for sentence_dict in sentences:
            sentence_id = sentence_dict["sentence_id"]
            sentence = sentence_dict["sentence"]

            # sanity check: do we have a label for each literal?
            for lit in sentence:
                if lit[0] == "w" or lit[0] == "prew":
                    littype, labelindex, dref = lit
                    if labelindex is None:
                        raise Exception("no label in word literal " + str(lit) + " " + filename)

            # sanity check: do we have a unary constraint for each
            # discourse referent mentioned in the sentence?
            sentence_okay = True

            drefs_with_unary_constraints = set(lit[2] for lit in sentence if lit[0] == "w" or lit[0] == "prew")
            
            for lit in sentence:
                if lit[0] == "r" or lit[0] == "prer":
                    # role
                    if len(lit) != 4:
                        print("Sentence with malformed literal, skipping:", sentence_id, lit)
                        malformed_sents += 1
                        sentence_okay = False
                        break
                    if lit[1] not in self.param_selpref:
                        print("Unknown role label, skipping:", sentence_id, lit)
                        malformed_sents += 1
                        sentence_okay = False
                        break
                    if lit[2] not in drefs_with_unary_constraints or lit[3] not in drefs_with_unary_constraints:
                        print("Discourse referent without unary constraint, skipping:", sentence_id, lit)
                        malformed_sents += 1
                        sentence_okay = False
                        break

                    
                elif lit[0] == "w" or lit[0] == "prew":
                    # word
                    if len(lit) != 3:
                        print("Sentence with malformed literal, skipping:",sentence_id, lit)
                        sentence_okay = False
                        malformed_sents += 1
                        break

                        malformed_sents += 1
                        sentence_okay = False
                        break
                                        
                else:
                    print("Sentence with malformed literal, skipping:", sentence_id, lit)
                    sentence_okay = False
                    malformed_sents += 1
                    break

            # santiy check: any reflexive relations?
            # if so, skip sentence
            prev_predarg_pairs = set()
            for lit in sentence:
                if lit[0] == "r":
                    _, _, pred, arg = lit
                    if (pred, arg) in prev_predarg_pairs:
                        sentence_okay = False
                        reflexive_sents += 1
                        break
                    prev_predarg_pairs.add( (pred, arg) )

            if sentence_okay: yield (sentence_id, sentence)

        if(verbose):
            print("Number of malformed sentences:", malformed_sents)
            print("Skipped sentences with reflexive relation:", reflexive_sents)

    ########################3
    # given a sentence, and the library of preevious entities, 
    # make a factor graph for it
    def build_factor_graph(self, sentence, mentalfiles):
        ######
        # factor graph wrapper
        fg =  MyFactorGraphWrapper()

        # number of concept nodes, which is the same as
        # number of scenario nodes
        wordliterals = [c for c in sentence if c[0] == "w" or c[0] =="prew"]
        roleliterals = [c for c in sentence if c[0] == "r" or c[0] == "prer"]
        presup_wordliterals = [c for c in sentence if c[0] =="prew"]
        presup_roleliterals = [c for c in sentence if c[0] == "prer"]
        
        num_nodes = len(wordliterals) 

        # make concept and scenario variables
        concept_variables, scenario_variables, = self.make_concept_and_scenario_vars(fg, num_nodes)

        ###
        # constraining word sense within this sentence
        
        # for each concept, a factor that restricts its values according to the word
        # returns: conceptvar_concepts: for each concept variable, list of valid concepts. overall, list of lists
        # conceptvar_indices: for each concept variable, its discourse referent. overall, a list
        conceptvar_concepts, conceptvar_indices = self.constrain_concepts_bywords(fg, concept_variables, wordliterals)

        # for each scenario/concept pair, a factor that restricts co-occurrences
        self.constrain_scenarios_concepts(fg, conceptvar_concepts, concept_variables, scenario_variables)

        # for each role, a factor that implements selectional constraints
        self.constrain_roles(fg, concept_variables, conceptvar_concepts, conceptvar_indices, roleliterals)

        # constrain scenarios according to a Dirichlet Multinomial
        # (approximately)
        self.constrain_scenarios(fg, scenario_variables, num_nodes, conceptvar_concepts, wordliterals)

        ###
        # adding variables and factors to deal with discourse referents
        # in a multi-sentence discourse
        
        ##
        # first, handling non-presupposed conditions 
        
        # make mental file variable to match any non-presupposed unary statements
        mf_variables, dref_concepts_mf = self.make_mentalfile_vars(fg, wordliterals, conceptvar_concepts)

        self.mentalfile_info = dref_concepts_mf

        # possibly add factors constraining co-occurrence of concepts in a mental file
        self.constrain_mentalfile_variables(fg, mf_variables, dref_concepts_mf)
        
        # link mental file variables to their matching concept variables
        self.constrain_concepts_mfvars(fg, wordliterals, mf_variables, concept_variables, dref_concepts_mf, conceptvar_concepts)

        ##
        # now, handle presupposed statements of the form "the X"
        
        # make index variables for all discourse referents in presupposed statements
        index_variables = self.make_index_vars(fg, presup_wordliterals, mentalfiles)
        
        # constrain concepts in unary presupposed statements to match the mental files
        self.constrain_coref_concepts(fg, presup_wordliterals, index_variables, concept_variables, conceptvar_concepts, mentalfiles)

        # for binary presupposed statements, constrain index variables to have parallel role linking
        self.constrain_coref_roles(fg, index_variables, presup_roleliterals, mentalfiles)

        # whenever we have some presupposed statements and some non-presupposed ones
        # for the same discourse referent, add factors to make sure mental file concept variables match
        self.constrain_coref_noncoref(fg, wordliterals, index_variables, concept_variables, conceptvar_concepts, mentalfiles)

        ###
        # finalize graph
        fg.make()

        return fg

    ##
    # make mental files random variables: binary variables
    # "is this concept true of this entity yes/no"
    # for each discourse referent that is introduced as new in the utterance,
    # make one such variable for each concept that, according to the sentence,
    # may apply to the referent
    def make_mentalfile_vars(self, fg, wordliterals, conceptvar_concepts):

        # for each discourse referent: which concepts are stated of it?
        # dref -> set of concepts
        dref_concepts = defaultdict(set)
        
        for i, literal in enumerate(wordliterals):
            # word type "w" or "prew", and discourse referent
            wtype, _, dref = literal

            if wtype != "w":
                # don't make a provisional new mental file entry
                # for discourse referents in statements that are presupposed
                continue

            dref_concepts[dref].update(conceptvar_concepts[i])


        # how many variables do we need?
        # for each discourse referent and each concept that the sentence
        # might have stated of the discourse referent, one variable
        num_mentalfile_vars = sum([len(v) for v in dref_concepts.values()])

        # make the variables
        mentalfile_variables = fg.add_variable_group(groupname = "mfconcept", num_vars = num_mentalfile_vars, num_states = 2)

        # return as mapping discourse referent -> concept ID -> mental files variable index
        retv = { }
        mfvar_index = 0
        for dref, concepts in dref_concepts.items():
            
            retv[dref] = { }
            
            for concept_id in concepts:
                retv[dref][concept_id] = mfvar_index
                mfvar_index += 1


        return mentalfile_variables, retv

    ##
    # add factors between concepts in the mental files: compatibility of concepts
    def constrain_mentalfile_variables(self, fg, mf_variables, dref_concepts_mf):
        pass

    ##
    # add factors linking concepts underlying content words in the utterance
    # to the matching mental files variables
    def constrain_concepts_mfvars(self, fg, wordliterals, mf_variables, concept_variables, dref_concepts_mf, conceptvar_concepts):

        
        for cix, literal in enumerate(wordliterals):        
            # word type "w" or "prew", and discourse referent
            wtype, _, dref = literal

            if wtype != "w":
                # we haven't made any mental file entry for thid literal
                continue            
            
            for concept_id in conceptvar_concepts[cix]:
                mfvar_ix = dref_concepts_mf[dref][concept_id]

                valid_configs = np.array([[c, int(c == concept_id)] for c in conceptvar_concepts[cix]])
                
                fg.add_factor(variables = [concept_variables[cix], mf_variables[mfvar_ix]],
                              factor_configs = valid_configs)
                
                
    
    ##
    # make index variables:
    # for each discourse referent that the utterance marks as coreferent
    # with an entity in the mental files,
    # make a random variable to indicate which mental files entity
    # it is coreferent with.
    # possible values: any entity index in the mental files
    def make_index_vars(self, fg, coref_drefs, mentalfiles):
        return [ ]


    ##
    # constrain index variables and concept variables for discourse referents that are coreferent
    # with an entity in the mental files:
    # concept predicated of a discourse referent in the current sentence
    # must match the concepts we know to apply or not apply to the coreferent entity according to the mental files
    def constrain_coref_concepts(self, fg, index_variables, concept_variables, conceptvar_indices, conceptvar_concepts, mentalfiles):
        pass

    ##
    # constrain index variables for pairs of discourse referents that are coreferent
    # with entities in the mental files:
    # roles said to link the two discourse referents in the current utterance
    # must match what we know of them in the entity files
    def constrain_coref_roles(self, fg, index_variables, roleliterals, mentalfiles):
        pass

    ##
    # when we have some non-presupposed statements and some presupposed statements about the same discourse referent,
    # then we have made some provisional mental file variables that may later need to be merged with an existing entry.
    # add factors to make sure there is no inconsistency
    def constrain_coref_noncoref(self, fg, wordliterals, index_variables, concept_variables, conceptvar_concepts, mentalfiles):
        pass
    
    ###########3
    # given a MAP result, extract mental file info
    # return a pair (nodes, edges) where nodes is a list of mental file entries
    # and edges is a list of pairs (index1, index2) of node indices
    # a mental file entry is a list of pairs (conceptid, value)
    #
    # HIER when we add this to an existing mental file, edge labels need to take into account
    # indices previously used
    def extend_mentalfiles_map(self, mapresult, sentence_id, sentence, mentalfiles):
        nodes, edges = mentalfiles

        for literal in sentence:
            if literal[0] == "w":
                # we have an entry in the new provisional mental files for this literal
                dref = literal[2]
                # mental file info is a mapping
                # discourse referent -> concept ID -> mental file variable index

                node = {"dref": [sentence_id, dref], "entry": [ [conceptid, mapresult["mfconcept"][mfix]] for conceptid, mfix in self.mentalfile_info[dref].items()] }

                nodes.append(node)

        for literal in sentence:
            if literal[0] == "r":
                # new edge, to be added to the mental files
                _, role, drefh, drefd = literal
                # HIER should we store the predicate name too, to have detailed roles? 
                edges.append( [[sentence_id, drefh], [sentence_id, drefd]])
                

        return [nodes, edges]

###################3

def onediscourse_map(sentences, sds_obj):
    mentalfiles = [[ ], []]
    mapresults = [ ]
    
    for sentence_id, sentence in sentences:
        fg = sds_obj.build_factor_graph(sentence, mentalfiles)

        # do the inference:
        # max-product algorithm
        try:
            # do the inference
            thismap = fg.map_inference()

            # store the MAP result
            mapresults.append(thismap)
        except Exception:
            print("Error in processing sentence, skipping:", sentence_id)

        # store new entries to the mental files
        mentalfiles =  sds_obj.extend_mentalfiles_map(thismap, sentence_id, sentence, mentalfiles) 

    return (mapresults, mentalfiles)
