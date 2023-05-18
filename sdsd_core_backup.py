# Situation description systems, with pgmax:
# discourse variant, every discourse referent
# has a quantifier, either "the" or "a"
#
# Data:
# list of utterance statements so far,
# keep dictionary discourse referent indices, mapped to word predicated of it
#    (there can be only one for now),
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

import sds_core

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

###########################
# helper class that runs posterior sampling
# over topics for a given sentence
class ScenarioSampler:
    # concept scenario prob: concept -> "scenario" : list of scenarios, "weight" -> list of log probabilities
    # alpha: dirichlet alpha
    # num topics: number of topics
    # num samples: how many samples to draw per Gibbs sampler run
    # num starts: how many runs/ restarts to do for the Gibbs sampler
    # discard: how many samples to discard at the beginning 
    def __init__(self, concept_scenario_logprob, word_concept_logprob, alpha, num_topics, num_words, num_samples, num_starts, discard):        
        # matrix: one row per topic, one column per word
        # probabilities, not log probabilities
        self.topic_word_prob = np.zeros((num_topics, num_words))

        for word_i in range(num_words):
            if str(word_i) in word_concept_logprob:
                concepts_for_i = list(word_concept_logprob[str(word_i)].items())
            else:
                concepts_for_i = [(str(word_i), 0.0)]

            for concept_j, wordi_conceptj_logprob in concepts_for_i:
                for scenario_k, scenk_concj_logprob in zip(concept_scenario_logprob[concept_j]["scenario"], concept_scenario_logprob[concept_j]["weight"]):
                    self.topic_word_prob[scenario_k, word_i] += math.exp(wordi_conceptj_logprob) * math.exp(scenk_concj_logprob)
        

        
        self.num_topics = num_topics        
        self.alpha = alpha
        self.num_samples = num_samples
        self.num_starts = num_starts
        self.discard = discard

    # run self.num_starts runs of a Gibbs sampler,
    # each time collecting self.num_samples samples
    # and discarding the first self.discard
    def sample_scenarios(self, sentence):
        
        results = [ ]
        for i in range(self.num_starts):
            initial = self._initialize(len(sentence))
            sample = self._gibbs_onerun(sentence, initial, self.num_samples)
            results.append(sample[self.discard:, :])

        samples = np.concatenate(results)

        return self._sample2logprobs(samples, len(sentence), sentence)


    # draw initial sample of topic assignments
    def _initialize(self,sentlen):
        return np.random.randint(0, self.num_topics, size = sentlen)

    # gibbs sampler, one run for given initial assignment
    def _gibbs_onerun(self, sentence, initial_assignment, num_samples):
        
        samples = np.empty([num_samples+1, len(sentence)], dtype = np.int64)  #sampled points
        samples[0] = initial_assignment

        for sample_ix in range(num_samples):

            thissample = [ ]
            # each "word" in this sentence is a list of concepts that could apply.
            # for each of them, sample a topic
            for i, word in enumerate(sentence):
                thissample.append(self._cond_sampler(word, samples[sample_ix], i))

            samples[sample_ix+1] = np.array(thissample)

        return samples
    
    # conditional sampler: resample topic for given word at focus index,
    # given all other topics
    def _cond_sampler(self, word, topics, focus_ix):
        # how often is each topic assigned, outside the focus index?
        topic_counts = self._count_topics_excluding(topics, focus_ix)

        # add the Dirichlet alpha
        topic_counts = topic_counts + self.alpha
        # print("HIER topic counts", topic_counts)

        # probability of focus word under all topics 
        focusword_probs = self.topic_word_prob[:, word]
        # print("HIER focus word probs", focusword_probs)

        # probability of topic z for this word right now:
        # focusword_probs[z] * (count_z + alpha) / (sum_z' count_z' + sentlength * alpha)
        topic_probs = np.array([focusword_probs[i] * topic_counts[i] for i in range(self.num_topics)])
        topic_probs = topic_probs / topic_probs.sum()
        # print("HIER topic probs", topic_probs)

        # re-sample a topic for this word
        return  np.random.choice(self.num_topics, p = topic_probs)

    # count how often each topic occurs in the sentence,
    # excluding one particular word
    def _count_topics_excluding(self, topics, exclude_ix):
        nonfocus_topics = np.array([t for i, t in enumerate(topics) if i != exclude_ix])
    
        return np.bincount(nonfocus_topics, minlength = self.num_topics)

    
    # turn samples into scenario probabilities.
    # return: for each word, a list of scenarios with nonzero probabilities, and log probabilities
    def _sample2logprobs(self, samples, numwords, sentence):
        word_topics_and_logprobs = [ ]
        for i in range(numwords):
            counts = np.bincount(samples[:, i], minlength = self.num_topics)
            scenarios_with_nonzero_counts = np.nonzero(counts)
            logprobs = np.log(counts[ scenarios_with_nonzero_counts] / sum(counts))
            word_topics_and_logprobs.append( (scenarios_with_nonzero_counts[0], logprobs))

        return word_topics_and_logprobs


    
################################
################################
class SDS:
    def __init__(self, vgpath_obj, scenario_config):
        self.scenario_handling = scenario_config["InSDS"]
        if self.scenario_handling not in ["tiled", "unary"]:
            print("Error: scenario handling method must be either 'tiled' or 'unary', I got:", self.scenario_handling)
            sys.exit(1)
            
        print("Scenario handling:", self.scenario_handling)
        
        self.tilesize = int(scenario_config["Tilesize"])
        self.tileovl = int(scenario_config["Tileoverlap"])
        self.top_scenarios_per_concept = int(scenario_config["TopScenarios"])
        
        self.vgpath_obj = vgpath_obj

        self.param_general, self.param_selpref, self.param_scenario_concept, self.param_word_concept = self.read_parameters(vgpath_obj)

        if self.scenario_handling == "tiled":
            # keep previously computed Dirichlet Multinomial log probabilities
            self.dirmult_obj = DirMultStore(self.param_general["dirichlet_alpha"], self.param_general["num_scenarios"])
        else:
            # make object for approximating the scenario distribution for a sentence
            # using a Gibbs sampler, and then sampling scenarios from the marginals
            self.scen_sampling_obj = ScenarioSampler(self.param_scenario_concept, self.param_word_concept, 
                                                     self.param_general["dirichlet_alpha"], self.param_general["num_scenarios"], self.param_general["num_words"],
                                                     int(scenario_config["NumSamples"]), int(scenario_config["Restarts"]), int(scenario_config["Discard"]))
                                                    
        
        
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
    #    ("a", discourse referent)
    #    ("the", discourse referent)
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
            # discourse referent mentioned in the sentence?
            # and do we have a quantifier for each discourse referent?
            sentence_okay = True

            drefs_with_unary_constraints = set(lit[2] for lit in sentence if lit[0] == "w")
            drefs_with_quantifiers = set(lit[1] for lit in sentence if lit[0] == "a" or lit[0] == "the")
            
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

                    if lit[2] not in drefs_with_quantifiers or lit[3] not in drefs_with_quantifiers:
                        print("Discourse referent without quantifier, skipping", sentence_id, lit)
                        malformed_sents += 1
                        sentence_okay = False
                        break
                        
                    
                elif lit[0] == "w":
                    # word
                    if len(lit) != 3:
                        print("Sentence with malformed literal, skipping:",sentence_id, lit)
                        sentence_okay = False
                        malformed_sents += 1
                        break

                    if lit[2] not in drefs_with_quantifiers:
                        print("Discourse referent without quantifier, skipping", sentence_idf, lit)
                        
                        malformed_sents += 1
                        sentence_okay = False
                        break
                    
                elif lit[0] in ["a", "the"]:
                    # quantifier
                    if len(lit) != 2:
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
    # given a sentence, and the library of preevious entities, 
    # make a factor graph for it
    def build_factor_graph(self, sentence, mentalfiles):
        ######
        # factor graph wrapper
        fg =  MyFactorGraphWrapper()

        # number of concept nodes, which is the same as
        # number of scenario nodes
        wordliterals = [c for c in sentence if c[0] == "w"]
        roleliterals = [c for c in sentence if c[0] == "r"]
        new_drefs = [c[1] for c in sentence if c[0] == "a"]
        coref_drefs = [c[1] for c in sentence if c[0] == "the"]
        
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
        # adding variables nd factors to deal with discourse referents
        # in a multi-sentence discourse
        
        ##
        # first, handling discourse-new discourse referents
        
        # make mental file variables for new discourse referents
        mf_variables, dref_concepts_mf = self.make_mentalfile_vars(fg, new_drefs, conceptvar_indices, conceptvar_concepts)

        self.mentalfile_info = dref_concepts_mf

        # possibly add factors constraining co-occurrence of concepts in a mental file
        self.constrain_mentalfile_variables(fg, mf_variables, dref_concepts_mf)
        
        # link mental file variables to their matching concept variables
        self.constrain_concepts_mfvars(fg, mf_variables, concept_variables, dref_concepts_mf, conceptvar_indices, conceptvar_concepts)

        ##
        # now, handle discourse referents marked as coreferent with an entity in the mental files
        
        # make index variables for coreferential discourse referents
        index_variables = self.make_index_vars(fg, coref_drefs, mentalfiles)
        
        # constrain concepts stated of coreferential discourse referents to match the mental files
        self.constrain_coref_concepts(fg, index_variables, concept_variables, conceptvar_indices, conceptvar_concepts, mentalfiles)

        # constrain index variables of corefeential discourse referents to have parallel role linking
        self.constrain_coref_roles(fg, index_variables, roleliterals, mentalfiles)
    

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
    def constrain_scenarios(self, fg, scenario_variables, num_nodes, conceptvar_concepts, wordliterals):

        # only one scenario? nothing to constrain
        if self.param_general["num_scenarios"] < 2:
            return

        if self.scenario_handling == "tiled":
            self.constrain_scenarios_tiled(fg, scenario_variables, num_nodes, conceptvar_concepts)

        elif self.scenario_handling == "unary":
            self.constrain_scenarios_unary(fg, scenario_variables, num_nodes, wordliterals)

        else:
            raise Exception("Unknown scenario handling " + str(self.scenario_handling))

    # instead of a single factor constraining all scenario nodes,
    # have "tiled" factors, each jointly restricting k scenario nodes, with some overlap.
    # this also assumes that we only have a limited number of scenarios per scenario node.
    def constrain_scenarios_tiled(self, fg, scenario_variables, num_nodes, conceptvar_concepts):
        
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

              
    # instead of a single factor constraining all scenario nodes,
    # run a sampler to determine posterior probabilities of scenarios for all concepts.
    # then add a unary factor for each scenario node restricting it according to its
    # marginal scenario probabilities
    def constrain_scenarios_unary(self, fg, scenario_variables, num_nodes, wordliterals):
        word_scenario_logprobs = self.scen_sampling_obj.sample_scenarios([ int(wordindex) for _, wordindex, _ in wordliterals])

        for i in range(num_nodes):
            sc_indices, logprobs = word_scenario_logprobs[i]
            
            fg.add_factor(variables = [ scenario_variables[i]],
                          factor_configs = np.array([[scx] for scx in sc_indices]),
                          log_potentials = logprobs )

    ##
    # make mental files random variables: binary variables
    # "is this concept true of this entity yes/no"
    # for each discourse referent that is introduced as new in the utterance,
    # make one such variable for each concept that, according to the sentence,
    # may apply to the referent
    def make_mentalfile_vars(self, fg, new_drefs, conceptvar_drefs, conceptvar_concepts):
        
        # make a mapping from discourse referent to
        # a set of concepts: for all the concept variables that
        # are about this discourse referent, all applicable concepts
        dref_concepts = defaultdict(set)
        for conceptid, dref in enumerate(conceptvar_drefs):
            dref_concepts[dref].update( conceptvar_concepts[ conceptid])


        # how many variables do we need?
        # for each discourse referent and each concept that the sentence
        # might have stated of the discourse referent, one variable
        num_mentalfile_vars = sum([len(v) for v in dref_concepts.values()])

        # make the variables
        mentalfile_variables = fg.add_variable_group(groupname = "mentalfile_con", num_vars = num_mentalfile_vars, num_states = 2)

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
    def constrain_concepts_mfvars(self, fg, mf_variables, concept_variables, dref_concepts_mf, conceptvar_indices, conceptvar_concepts):
        
        for cix, conceptvar in enumerate(concept_variables):
            
            dref = conceptvar_indices[cix]
            for concept_id in conceptvar_concepts[cix]:
                mfvar_ix = dref_concepts_mf[dref][concept_id]

                valid_configs = np.array([[c, int(c == concept_id)] for c in conceptvar_concepts[cix]])
                
                fg.add_factor(variables = [conceptvar, mf_variables[mfvar_ix]],
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

###################3

def onediscourse_map(sentences, sds_obj):
    mentalfiles = [ ]
    mapresults = [ ]
    
    for sentence_id, sentence in sentences:
        fg = sds_obj.build_factor_graph(sentence, mentalfiles)

        # do the inference:
        # max-product algorithm
        try:
            thismap = fg.map_inference()
            mapresults.append(thismap)
            # HIER add something to the mental files
        except Exception:
            print("Error in processing sentence, skipping:", sentence_id)


    return (mapresults, mentalfiles)
