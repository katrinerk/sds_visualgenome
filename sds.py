# Situation description systems, with pgmax:
#
# Multiple mentions of individuals.
# No incremental computation.
# one scenario/concept per unary condition.
#
# when setting different prior probabilities for scenarios,
# give them in alphabetical ordering of scenario names
#
# Data:
# list of utterance statements so far,
# keep dictionary discourse referent indices, mapped to word predicated of it
#    (there can be only one for now),
#
#

# What to do about concepts that aren't connected to any scenarios?
# if all concepts for a word aren't connected to scenarios, we want to not
# have a scenario node -- and pass that information on.
# but what if some of the concepts of a node aren't connected to scenarios?
# is their value unconstrained?

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


#######################3
# helper class that stores Dirichlet-Multinomial values
class DirMultStore:
    def __init__(self, alpha, num_scenarios):
        self.alpha = alpha
        self.num_scenarios = num_scenarios
        self.store = { }

    def logps(self, valid_configs):
        # 2-dimensional bin count to count how often each value appears
        # in each valid configuration.
        # then sort the resulting bin counts because we assume that we have a
        # symmetric alpha, so it doesn't matter which value is which
        bincounts = np.sort(self._bincount2d(valid_configs, self.num_scenarios))

        retv = [ ]
        for bincount in bincounts:
            tb = tuple(bincount)
            if tb in self.store: retv.append(self.store[tb])
            else:
                val = logddirmult(bincount, self.alpha)
                self.store[tb] = val
                retv.append(val)

        return retv

    # from https://stackoverflow.com/questions/19201972/can-numpy-bincount-work-with-2d-arrays
    def _bincount2d(self, arr, bins=None):
        if bins is None:
            bins = np.max(arr) + 1
        count = np.zeros(shape=[len(arr), bins], dtype=np.int64)
        indexing = (np.ones_like(arr).T * np.arange(len(arr))).T
        np.add.at(count, (indexing, arr), 1)

        return count

################################
################################
class SDS:
    def __init__(self, vgpath_obj, scenario_config):
        self.scenario_handling = scenario_config["InSDS"]
        print("Scenario handling:", self.scenario_handling)
        
        if self.scenario_handling != "tiled":
            print("Only tiled handling of scenarios implemented so far")
            sys.exit(1)

        self.tilesize = int(scenario_config["Tilesize"])
        self.tileovl = int(scenario_config["Tileoverlap"])
        self.top_scenarios_per_concept = int(scenario_config["TopScenarios"])
        
        self.vgpath_obj = vgpath_obj

        self.param_general, self.param_selpref, self.param_scenario_concept, self.param_word_concept = self.read_parameters(vgpath_obj)

        # keep previously computed Dirichlet Multinomial log probabilities
        self.dirmult_obj = DirMultStore(self.param_general["dirichlet_alpha"], self.param_general["num_scenarios"])

        
        
    ########################3
    # read global parameters,
    # scenario/concept constraints,
    # selectional preferences.
    # return as one big dictionary
    def read_parameters(self, vgpath_obj):

        filenames = vgpath_obj.sds_filenames()

        # read general info
        with open(filenames["general"]) as f:
            param_general = json.load(f)

        # read selectional constraints
        selpref_zipfile, selpref_file = filenames["selpref"]     
        with zipfile.ZipFile(selpref_zipfile) as azip:
            with azip.open(selpref_file) as f:
                param_selpref = json.load(f)

        # read scenario/concept weights
        scenario_zipfile, scenario_file = filenames["scenario_concept"]
        with zipfile.ZipFile(scenario_zipfile) as azip:
            with azip.open(scenario_file) as f:
                prelim_param_scenario_concept = json.load(f)

        if self.scenario_handling == "tiled":
            # restrict each concept to its top n scenarios
            print("Restricting each concept to its top", self.top_scenarios_per_concept, "scenarios")
            param_scenario_concept = { }
            for conceptindex in prelim_param_scenario_concept.keys():
                top_n_scenarios_and_weights = sorted( zip( prelim_param_scenario_concept[conceptindex]["scenario"], prelim_param_scenario_concept[conceptindex]["weight"]),
                                                           key = lambda pair:pair[1], reverse = True)[:self.top_scenarios_per_concept]
                
                param_scenario_concept[ conceptindex ] = { "scenario" : [s for s, w in top_n_scenarios_and_weights],
                                                           "weight" : [w for s, w in top_n_scenarios_and_weights]}
        else:
            # nothing to be changed, keep all scenarios for each concept
            param_scenario_concept = prelim_param_scenario_concept

        # read word/concept weights, if any
        wordconcept_zipfile, wordconcept_file = filenames["word_concept"]
        if os.path.isfile(wordconcept_zipfile):
            with zipfile.ZipFile(wordconcept_zipfile) as azip:
                with azip.open(wordconcept_file) as f:
                    param_word_concept = json.load(f)
        else:
            param_word_concept = { }

        return ( param_general, param_selpref, param_scenario_concept, param_word_concept )

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
    #    ("r", "arg1", pred discourse referent, argument 1 discourse referent)
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
                if lit[0] == "w":
                    littype, labelindex, dref = lit
                    if labelindex is None:
                        raise Exception("no label in word literal " + str(lit) + " " + filename)

            # sanity check: do we have a unary constraint for each
            # discourse referent mentioned in the sentence
            sentence_okay = True

            drefs_with_unary_constraints = set(lit[2] for lit in sentence if lit[0] == "w")
            for lit in sentence:
                if lit[0] == "r":
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
                elif lit[0] == "w":
                    if len(lit) != 3:
                        print("Sentence with malformed literal, skipping:",sentence_id, lit)
                        sentence_okay = False
                        malformed_sents += 1
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
    # given a sentence,
    # make a factor graph for it
    def build_factor_graph(self, sentence):
        ######
        # factor graph wrapper
        fg =  MyFactorGraphWrapper()

        # number of concept nodes, which is the same as
        # number of scenario nodes
        wordliterals = [c for c in sentence if c[0] == "w"]
        roleliterals = [c for c in sentence if c[0] == "r"]
        num_nodes = len(wordliterals)

        # make concept and scenario variables
        concept_variables, scenario_variables, = self.make_concept_and_scenario_vars(fg, num_nodes)

        # for each concept, a factor that restricts its values according to the word
        conceptvar_concepts, conceptvar_indices = self.constrain_concepts_bywords(fg, concept_variables, wordliterals)

        # for each scenario/concept pair, a factor that restricts co-occurrences
        self.constrain_scenarios_concepts(fg, conceptvar_concepts, concept_variables, scenario_variables)

        # for each role, a factor that implements selectional constraints
        self.constrain_roles(fg, concept_variables, conceptvar_concepts, conceptvar_indices, roleliterals)

        # across all scenarios, a factor (or constellation of factors)
        # to implement the Dirichlet Multinomial distribution
        self.constrain_scenarios(fg, scenario_variables, num_nodes, conceptvar_concepts)

        ###
        # finalize graph
        fg.make()

        return fg

    ##########
    # making the main variables
    def make_concept_and_scenario_vars(self, fg, num_nodes):

        scenario_variables = fg.add_variable_group(groupname = "scenario", num_vars = num_nodes, num_states = self.param_general["num_scenarios"])
        concept_variables = fg.add_variable_group(groupname = "concept", num_vars = num_nodes, num_states = self.param_general["num_concepts"])

        return (concept_variables, scenario_variables)


    ##########
    # for each concept, restrict its values according to the corresponding word
    def constrain_concepts_bywords(self, fg, concept_variables, wordliterals):
        # returns:
        # for each concept variable, list of valid concepts
        conceptvar_concepts = [ ]
        # for each concept variable, its discourse referent
        conceptvar_dref = [ ]

        for i, conceptvariable in enumerate(concept_variables):
            # take apart the word
            _, wordindex, discoursereferent = wordliterals[i]

            # do we have concepts listed for this word?
            # if not, take concept index to be word index
            wordkey = str(wordindex)
            if wordkey in self.param_word_concept:
                # we do have an entry for this word.
                # concept indices may differ from the word index
                concept_indices = [int(c) for c in self.param_word_concept[wordkey].keys()]

                # store concept indices for this word
                conceptvar_concepts.append( concept_indices)

                # make factor.
                # configurations: all relevant concept indices.
                # log probabilities: as stored for these concept indices in the parameter dictionary
                fg.add_factor(variables = [ concept_variables[i] ],
                         factor_configs = np.array([ [c] for c in concept_indices], dtype = int),
                         log_potentials = np.array([ self.param_word_concept[wordkey][str(c)] for c in concept_indices], dtype = np.float64))

            else:
                # concept index is same as word index
                # no ambiguity
                conceptindex = wordindex

                # store the concept variable for this word as being concept index
                conceptvar_concepts.append( [ conceptindex ])

                # make factor: just one possible concept, with probability 1
                fg.add_factor(variables = [ concept_variables[i] ],
                         factor_configs = np.array([ [conceptindex] ], dtype = int))

            # store the discourse referent from the current literal
            # as going with this concept-valued variable
            conceptvar_dref.append( discoursereferent)

        return conceptvar_concepts, conceptvar_dref



    ##########
    # for a scenario/concept pair ,restrict co-occurrences
    def constrain_scenarios_concepts(self, fg, conceptvar_concepts, concept_variables, scenario_variables):
        num_nodes = len(conceptvar_concepts)


        # conceptvar-concepts: list of concept lists, one for each word in the sentence
        # transform into a set of relevant concepts
        relevant_concepts = set()
        for ccs in conceptvar_concepts: relevant_concepts.update(ccs)

        # valid configuration: [scenario i, concept j] for all relevant concepts
        valid_configs = [ ]
        logprobs = []
        for conceptindex in relevant_concepts:

            if str(conceptindex) not in self.param_scenario_concept:
                print("Not constraining scenario for concept", conceptindex)
                continue

            for scenarioindex, weight in zip(self.param_scenario_concept[str(conceptindex)]["scenario"], self.param_scenario_concept[str(conceptindex)]["weight"]):
                valid_configs.append( [scenarioindex, conceptindex] )
                logprobs.append(weight)

        fg.add_factor_group(variable_groups = [ [scenario_variables[i], concept_variables[i]] for i in range(num_nodes)],
                            factor_configs = np.array(valid_configs, dtype = int),
                            log_potentials = np.array(logprobs, dtype = np.float64))

        # print("constraint-scenario factors on", [ [scenario_variables[i], concept_variables[i]] for i in range(num_nodes)])



    ##########
    # add a factor restricting the concepts at dependent-index
    def constrain_roles(self, fg, concept_variables, conceptvar_concepts, conceptvar_dref, roleliterals):

        # flatten conceptvar_concepts into a single set
        concepts_this_sent = set(item for sublist in conceptvar_concepts for item in sublist)
        
        # for every type of role, add factor group
        for roletype in self.param_selpref.keys():
            # collect pairs of concept index for predicate, concept index for argument.
            variable_pairs = [ ]
            for _, rtype, headdref, depdref in roleliterals:
                if rtype != roletype: continue

                # there could be several concepts associated with this head discourse referent/
                # modifier discourse referent.
                # connect every pair of such concepts
                headconceptix = [i for i, d in enumerate(conceptvar_dref) if d == headdref]
                depconceptix = [i for i, d in enumerate(conceptvar_dref) if d == depdref]

                for hix in headconceptix:
                    for dix in depconceptix:
                        variable_pairs.append( (hix, dix))

            if len(variable_pairs) > 0:
                factor_configs = [ ]
                log_potentials = [ ]
                for config, logp in zip(self.param_selpref[roletype]["config"], self.param_selpref[roletype]["weight"]):
                    if config[0] in concepts_this_sent and config[1] in concepts_this_sent:
                        factor_configs.append(config)
                        log_potentials.append(logp)

                
                fg.add_factor_group(variable_groups = [ [concept_variables[i], concept_variables[j]]  for i, j in variable_pairs],
                                    factor_configs = np.array(factor_configs, dtype = int),
                                    log_potentials = np.array(log_potentials, dtype = np.float64))

                # fg.add_factor_group(variable_groups = [ [concept_variables[i], concept_variables[j]]  for i, j in variable_pairs],
                #                     factor_configs = np.array(self.param_selpref[roletype]["config"], dtype = int),
                #                     log_potentials = np.array(self.param_selpref[roletype]["weight"], dtype = np.float64))

                # print("role factor for role type", roletype, variable_pairs, [ [concept_variables[i], concept_variables[j]]  for i, j in variable_pairs])



    ##########
    # Dirichlet-Multinomial as constraint on co-occurrences of scenarios
    def constrain_scenarios(self, fg, scenario_variables, num_nodes, conceptvar_concepts):

        # only one scenario? nothing to constrain
        if self.param_general["num_scenarios"] < 2:
            return

        # "tiling" scenario constraints:
        # we are only constraining `self.tilesize` scenarios at a time,
        # with an overlap of `self.tileovl`
        tile_start_indices = list(range(0, num_nodes, self.tilesize - self.tileovl))
        if len(tile_start_indices) > 1 and num_nodes - tile_start_indices[-1] <= self.tileovl:
            # we have at least two tiles, but the last one would be shorter than the overlap
            # so it's completely covered by the next-to-last tile
            # so delete the last tile
            tile_start_indices = tile_start_indices[:-1]

        # print ("sentence length", num_nodes, "num tiles:", len(tile_start_indices))

        for tileno, startindex in enumerate(tile_start_indices):
            # determine the end index for this tile
            if tileno == len(tile_start_indices) -1:
                # last tile
                endindex = num_nodes - 1
            else:
                # not last tile
                endindex = startindex + self.tilesize

            # mutually constrain scenarios from startindex to endindex

            # valid configurations:
            # for each node, each concept that the node can take on,
            # any of the scenarios listed for it in param_scenario_concept

            # for each node, in order, make list of all scenarios it can take on
            node_scenarios = [ ]
            for concepts in conceptvar_concepts[startindex : endindex + 1]:
                node_scenarios.append(list(set(s for c in concepts for s in self.param_scenario_concept[str(c)]["scenario"])))

            valid_configs = np.array(list(itertools.product(*node_scenarios)))
            # and their log probablities:
            # first transform configurations to scenario counts, then hand those to logddirmult
            logps = self.dirmult_obj.logps(valid_configs)

            fg.add_factor(variables = [scenario_variables[i] for i in range(startindex, endindex + 1)],
                            factor_configs = valid_configs,
                            log_potentials = np.array(logps))

            # print("scenario factor", [scenario_variables[i] for i in range(num_nodes)])

              

    
######################3
def main():

    # command line arguments
    parser = ArgumentParser()
    parser.add_argument('input', help="directory with input sentences")
    parser.add_argument('output', help="directory for system output")    

    args = parser.parse_args()

    # settings file
    config = configparser.ConfigParser()
    config.read("settings.txt")

    vgpath_obj = VGPaths(sdsdata = args.input, sdsout = args.output)

    # make SDS object
    sds_obj = SDS(vgpath_obj, config["Scenarios"])

    # process each utterance go through utterances in the input file,
    sentlength_runtime = defaultdict(list)

    # counter for progress bar
    counter = 0

    # store MAP results and marginals
    results = [ ]
    
    for sentence_id, sentence in sds_obj.each_sentence_json(verbose = True):

        sentlength = sum(1 for ell in sentence if ell[0] == "w")

        counter += 1
        if counter % 20 == 0: print(counter)
        # if counter > 50: break

        # print("SENTENCE")
        # for ell in sentence: print(ell)
        # print()

        # make a timer to see how long it takes to do
        # graph creation + inference
        tic=timeit.default_timer()

        # construct factor graph
        fg = sds_obj.build_factor_graph(sentence)

        # do the inference:
        # max-product algorithm
        thismap = fg.map_inference()
        results.append({"sentence_id" : sentence_id, "MAP" : thismap})

        # stop timer
        toc=timeit.default_timer()
        sentlength_runtime[sentlength].append(toc - tic)

    # write results
    # into the output zip file
    zipfilename, filename = vgpath_obj.sds_output_zipfilename( write = True)
    with zipfile.ZipFile(zipfilename, "w", zipfile.ZIP_DEFLATED) as azip:
        azip.writestr(filename, json.dumps(results))

                

    for ell in sorted(sentlength_runtime.keys()):
        print("Sentence length", ell, "#sentences", len(sentlength_runtime[ell]),
              "mean runtime", round(statistics.mean(sentlength_runtime[ell]), 3))

if __name__ == '__main__':
    main()
