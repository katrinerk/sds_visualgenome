# Situation description systems, with pgmax:
# multi-sentence discourse
# can have presupposed material, labeled "prew" or "prer" for
# presupposed words or roles
#
# Current implementation:
# after a sentence has been processed, the mental files record the MAP concept assignments
# for each entity. Those assignments aren't changeable anymore later.
# Another possibility would be to record probabilistic concept assignments
# and make them updateable

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
    # read each group of sentences from zipped archive,
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
    def each_passage_json(self, verbose = True):
        inputzipfilename, filename = self.vgpath_obj.sds_sentence_zipfilename()
        with zipfile.ZipFile(inputzipfilename) as azip:
            with azip.open(filename) as f:
                passages = json.load(f)

        malformed_sents = 0
        reflexive_sents = 0

        # passage: a list of sentences
        for passage in passages:
            transformed_passage = [ ]
            passage_okay = True
            
            for sentence_dict in passage:
                result = self._sanitycheck_sentence(sentence_dict)

                if result.get("malformed_lit", False):
                    malformed_sents += 1
                    passage_okay = False
                elif result.get("reflexive", False):
                    reflexive_sents += 1
                    passage_okay = False

                else:
                    transformed_passage.append( (result["sentence_id"], result["sentence"]))

            if passage_okay:
                yield transformed_passage
            else:
                print("HIER passage not okay")

        if(verbose):
            print("Number of malformed sentences:", malformed_sents)
            print("Skipped sentences with reflexive relation:", reflexive_sents)


    ######
    # sanity check on a single sentence
    def _sanitycheck_sentence(self, sentence_dict):
        sentence_id = sentence_dict["sentence_id"]
        sentence = sentence_dict["sentence"]

        # print("Sentence", sentence_id, sentence)
        
        # sanity check: do we have a label for each literal?
        for lit in sentence:
            if lit[0] == "w" or lit[0] == "prew":
                littype, labelindex, dref = lit
                if labelindex is None:
                    raise Exception("no label in word literal " + str(lit) + " " + filename)

        # sanity check: do we have a unary constraint for each
        # discourse referent mentioned in the sentence?
        drefs_with_unary_constraints = set(lit[2] for lit in sentence if lit[0] == "w" or lit[0] == "prew")

        for lit in sentence:
            if lit[0] == "r" or lit[0] == "prer":
                # role
                if len(lit) != 4:
                    print("Sentence with malformed literal, skipping:", sentence_id, lit)
                    return {"sentence_id" : sentence_id, "sentence" : sentence, "malformed_lit" : True}
                
                if lit[1] not in self.param_selpref:
                    print("Unknown role label, skipping:", sentence_id, lit)
                    return {"sentence_id" : sentence_id, "sentence" : sentence, "malformed_lit" : True}

                if lit[2] not in drefs_with_unary_constraints or lit[3] not in drefs_with_unary_constraints:
                    print("Discourse referent without unary constraint, skipping:", sentence_id, lit)
                    return {"sentence_id" : sentence_id, "sentence" : sentence, "malformed_lit" : True}


            elif lit[0] == "w" or lit[0] == "prew":
                # word
                if len(lit) != 3:
                    print("Sentence with malformed literal, skipping:",sentence_id, lit)
                    return {"sentence_id" : sentence_id, "sentence" : sentence, "malformed_lit" : True}

            else:
                print("Sentence with malformed literal, skipping:", sentence_id, lit)
                return {"sentence_id" : sentence_id, "sentence" : sentence, "malformed_lit" : True}

        # santiy check: any reflexive relations?
        # if so, skip sentence
        prev_predarg_pairs = set()
        for lit in sentence:
            if lit[0] == "r":
                _, _, pred, arg = lit
                if (pred, arg) in prev_predarg_pairs:
                    return {"sentence_id" : sentence_id, "sentence" : sentence, "reflexive" : True}
                
                prev_predarg_pairs.add( (pred, arg) )
                
        return {"sentence_id" : sentence_id, "sentence" : sentence}
                    
        
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
        
        # make mental file variable to match any non-presupposed unary statements.
        # dref_concepts_mf: mapping discourse referent -> concept -> mental file variable index
        mf_variables, dref_concepts_mf = self.make_mentalfile_vars(fg, wordliterals, conceptvar_concepts)

        self.mentalfile_info = dref_concepts_mf

        # # add factors constraining co-occurrence of concepts in a mental file
        if mf_variables is not None:
            self.constrain_mentalfile_variables(fg, mf_variables, dref_concepts_mf)
        
            # link mental file variables to their matching concept variables
            self.constrain_concepts_mfvars(fg, wordliterals, mf_variables, concept_variables, dref_concepts_mf, conceptvar_concepts)

        ##
        # now, handle presupposed statements of the form "the X"
        
        # make index variables for all discourse referents in presupposed statements
        # dref_indexvar: mapping discourse referent -> index of index variable
        index_variables, dref_indexvar = self.make_index_vars(fg, wordliterals, roleliterals, mentalfiles)

        self.coref_info = dref_indexvar
        
        if index_variables is not None:
            # there are some presuppositions, some connections to the mental files to make
            
            # constrain concepts in unary presupposed statements to match the mental files
            # indexvar_mfindex; mapping index variable index -> list of possible mental file concept indices
            indexvar_mfindex = self.constrain_coref_concepts(fg, wordliterals, index_variables, concept_variables, dref_indexvar, conceptvar_concepts, mentalfiles)
            if indexvar_mfindex is None:
                # presupposition failure
                return None

            # for binary presupposed statements, constrain index variables to have parallel role linking
            result = self.constrain_coref_roles(fg, index_variables, roleliterals, dref_indexvar, indexvar_mfindex, mentalfiles)
            if result is None:
                # presupposition failure
                return None

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

        if num_mentalfile_vars == 0:
            # no new entities in this sentence
            return (None, { })

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
    def make_index_vars(self, fg, wordliterals, roleliterals, mentalfiles):
        
        # which discourse referents occur in presupposed unary or binary literals?
        # those are the ones that need an index variable
        drefs_in_presupposed_literals = set(dref for wtype, _, dref in wordliterals if wtype == "prew")
        drefs_in_presupposed_literals.update(drefh for rtype, _, drefh, _ in roleliterals if rtype == "prer")
        drefs_in_presupposed_literals.update(drefd for rtype, _, _, drefd in roleliterals if rtype == "prer")

        if len(drefs_in_presupposed_literals) == 0:
            # nothing presupposed
            return (None, { })

        num_indexvar_values = len(mentalfiles["entity"])
        # print("this many discourse refs in presupposed literals", len(drefs_in_presupposed_literals), "this many values", num_indexvar_values)
        
        index_variables = fg.add_variable_group(groupname = "drefindex", num_vars = len(drefs_in_presupposed_literals), num_states = num_indexvar_values)

        # make a mapping from discourse referent to index variable
        dref_indexvar = dict((d, i) for d, i in enumerate(list(drefs_in_presupposed_literals)))

        return (index_variables, dref_indexvar)
        


    ##
    # constrain index variables and concept variables for discourse referents that are coreferent
    # with an entity in the mental files:
    # concept predicated of a discourse referent in the current sentence
    # must match the concepts we know to apply or not apply to the coreferent entity according to the mental files
    def constrain_coref_concepts(self, fg, wordliterals, index_variables, concept_variables, dref_indexvar, conceptvar_concepts, mentalfiles):
        # keep track of which index variables could have which mental file indices as values
        ixvar_mfix = { }
        
        # for each unary literal marked as presupposed:
        # find entities in the mental files that the discourse referent could refer to
        for cix, ell in enumerate(wordliterals):
            
            if ell[0] != "prew":
                # not presupposed? nothing to do
                continue

            _, _, dref = ell
            
            # concepts that could underlie this content word
            concepts = conceptvar_concepts[cix]
            
            # keep all mental file indices where one of the concepts that the entity is probably an instance of (p>0)
            # is one of the concepts that could underlie this content word,
            # also keep, for each mental file index, the relevant concepts
            mfix_concepts = { }
            for i, entry in enumerate(mentalfiles["entity"]):
                overlap_concepts = [c for c, prob in entry["entry"] if c in concepts ]
                if len(overlap_concepts) > 0:
                    mfix_concepts[i] = overlap_concepts
                    

            if len(mfix_concepts) == 0:
                # there is no entity at all in the mental files that this discourse referent could refer to.
                # presupposition failed.
                print("Presupposed unary literal with no match in the mental files", ell)
                return None

            # there are possible values, make a factor
            
            # which is the index variable?
            df_index_ix = dref_indexvar[dref ]

            # store possible mental file indices for this index variable
            ixvar_mfix[ df_index_ix] = list(mfix_concepts.keys())
            
            # valid configurations:
            valid_configs = np.array([ [c, mfix]  for mfix, ovlconcepts in mfix_concepts.items() for c in ovlconcepts])

            
            fg.add_factor(variables = [concept_variables[cix], index_variables[df_index_ix]],
                          factor_configs = valid_configs)

        return ixvar_mfix
    ##
    # constrain index variables for pairs of discourse referents that are coreferent
    # with entities in the mental files:
    # roles said to link the two discourse referents in the current utterance
    # must match what we know of them in the entity files
    def constrain_coref_roles(self, fg, index_variables, roleliterals, dref_indexvar, indexvar_mfindex, mentalfiles):
        for rtype, role, drefh, drefd in roleliterals:
            if rtype != "prer":
                # not presupposed relation, nothing to do
                continue

            # which index variables are associated with the two discourse referents?
            indexh = dref_indexvar[ drefh]
            indexd = dref_indexvar[drefd]

            # which mental file discourse referents might be associated with the two index variables?
            # from indexh/d determine the list of mental file indices associated with indexh/d.
            # then make a mapping discourse referent -> mental file index

            mfdrefs_h = dict( (tuple(mentalfiles['entity'][i]["dref"]), i) for i in indexvar_mfindex[indexh])
            mfdrefs_d = dict( (tuple(mentalfiles['entity'][i]["dref"]), i) for i in indexvar_mfindex[indexd])
            

            # make a factor connecting the two index variables:
            # they can have a value combination i1, i2 if there is a role of the right type
            # recorded in the mental files between dref(i1) and dref(i2)
            candidate_mfindices = [ [mfdrefs_h[tuple(dh)], mfdrefs_d[tuple(dd)]] for dh, r, dd in mentalfiles["roles"] if r == role and tuple(dh) in mfdrefs_h and tuple(dd) in mfdrefs_d]
            
            if len(candidate_mfindices) == 0:
                # presupposition failure, no such known role
                print("Presupposed binary literal with no match in the mental files", role, drefh, drefd)
                return None

            fg.add_factor(variables = [index_variables[indexh], index_variables[indexd]],
                          factor_configs = np.array(candidate_mfindices))


        return True

            
    ##
    # when we have some non-presupposed statements and some presupposed statements about the same discourse referent,
    # then we have made some provisional mental file variables that may later need to be merged with an existing entry.
    # add factors to make sure there is no inconsistency
    def constrain_coref_noncoref(self, fg, wordliterals, index_variables, concept_variables, conceptvar_concepts, mentalfiles):
        pass
    
    ###########3
    # result read-off

    ##
    # given a MAP result, extract mental file info
    # return a pair (nodes, edges) where nodes is a list of mental file entries
    # a mental file entry is structure
    # { "dref" : (sentence ID, discourse referent), "entry": list of pairs (concept ID, value) }
    # 
    # and edges is a list of tuples (discourse refernet, role, discourse referent)
    # (again including sentence IDs)
    def extend_mentalfiles_map(self, mapresult, sentence_id, sentence, mentalfiles):
        nodes = mentalfiles["entity"]
        edges = mentalfiles["roles"]

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
                # store edge as tuple of discourse referents with role label:
                # because for each discourse referent that denotes an event or an attribute,
                # there will only ever by one unary predicate stated
                # so we don't have to keep track of which predicate-specific role links
                # drefd and drefh
                edges.append( [[sentence_id, drefh], role, [sentence_id, drefd]])
                

        return {"entity": nodes, "roles" : edges}

    ##
    # given a MAP result,
    # determine which discourse referent gets mapped to which mental files index
    def coref_readoff_map(self, mapresult):
        if "drefindex" not in mapresult:
            return { }
        
        return dict((dref, mapresult["drefindex"][indexvar]) for dref, indexvar in self.coref_info.items())

###################3

def onediscourse_map(sentences, sds_obj):
    mentalfiles = {"entity": [ ], "roles":[]}
    mapresults = [ ]
    corefresults = [ ]
    paragraph_okay = True
    
    for sentence_id, sentence in sentences:
        fg = sds_obj.build_factor_graph(sentence, mentalfiles)
        if fg is None:
            # factor graph not successfuly built
            print("Error in creating factor graph for sentence, skipping", sentence_id)
            paragraph_okay = False
            break

        # do the inference:
        # max-product algorithm
        try:
            thismap = fg.map_inference()

            # store the MAP result
            mapresults.append(thismap)
            
            # store new entries to the mental files
            mentalfiles =  sds_obj.extend_mentalfiles_map(thismap, sentence_id, sentence, mentalfiles)

            # did we determine any coreference?
            if "drefindex" in thismap:
                # yes: read off the mapping from discourse referent to mental file entry.
                
                # dref_mfindex: list of pairs (discourse referent, index into the mental files).
                # iterate over those pairs.
                dref_mfindex = sds_obj.coref_readoff_map(thismap)
                for dref, mfindex in dref_mfindex.items():
                    # entry: entity entry at this mental file index,
                    # dictionary:
                    #  "dref" : discourse referent
                    #  "entry": list of pairs ( concept index, probability)
                    entry = mentalfiles["entity"][mfindex]
                    corefresults.append( [ [sentence_id, dref],  entry["dref"],  [cix for cix, prob in entry["entry"] if prob > 0.0]] )

        except Exception:
            print("Error in running inference on sentence, skipping:", sentence_id)
            paragraph_okay = False


    if not paragraph_okay:
        return None
    else:
        return (mapresults, mentalfiles, corefresults)
