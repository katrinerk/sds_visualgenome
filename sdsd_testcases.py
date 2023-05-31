# Katrin Erk May 2023
# Write input for an SDS system:
# parameter files, and input sentences
#
# Here: multi-sentence discourse, with mental files.
# identify referent for a given NP


import sys
import os
import json
import zipfile
from collections import defaultdict, Counter
import math
import numpy as np
from argparse import ArgumentParser
import random
import configparser

from vgnames import VGOBJECTS, VGATTRIBUTES, VGRELATIONS 
import vgiterator
from sds_input_util import VGParam
from vgindex import VgitemIndex
from vgpaths import VGPaths
from polysemy_util import SyntheticPolysemes
from sdsd_core import SDSD
from sds_core import ScenarioSampler, DirMultStore
from sentence_util import SentencePrinter

########3




parser = ArgumentParser()
parser.add_argument('--sdsdata', help="directory for SDS parameters, default: sds_in/discourse", default = "sds_in/discourse/")
parser.add_argument('--vgdata', help="directory with VG data including frequent items, train/test split, topic model", default = "data/")

args = parser.parse_args()

vgpath_obj = VGPaths(vgdata = args.vgdata, sdsdata = args.sdsdata)

# settings file
config = configparser.ConfigParser()
config.read("settings.txt")
selpref_method = config["Selpref"]

print("reading data")

# frequent obj/attr/rel
vgcounts_zipfilename, vgcounts_filename = vgpath_obj.vg_counts_zip_and_filename()
with zipfile.ZipFile(vgcounts_zipfilename) as azip:
    with azip.open(vgcounts_filename) as f:
        vgobjects_attr_rel = json.load(f)
        
vgindex_obj = VgitemIndex(vgobjects_attr_rel)

out_obj = SentencePrinter(vgindex_obj)

####
# compute parameters for SDS.
print("computing parameters")
vgparam_obj = VGParam(vgpath_obj, selpref_method, frequentobj = vgobjects_attr_rel)


global_param, scenario_concept_param, word_concept_param, selpref_param = vgparam_obj.get()
vgparam_obj.write(global_param, scenario_concept_param, word_concept_param, selpref_param)

###
# adding polysemy?
class MySDSD(SDSD):
    def __init__(self, vgpath_obj, scenario_config):
        
        self.scenario_config = scenario_config
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

        # keep previously computed Dirichlet Multinomial log probabilities
        if self.scenario_handling == "tiled":
            self.dirmult_obj = DirMultStore(self.param_general["dirichlet_alpha"], self.param_general["num_scenarios"])
        else:
            self.scen_sampling_obj = ScenarioSampler(self.param_scenario_concept, self.param_word_concept, 
                                                     self.param_general["dirichlet_alpha"], self.param_general["num_scenarios"], self.param_general["num_words"],
                                                     int(scenario_config["NumSamples"]), int(scenario_config["Restarts"]), int(scenario_config["Discard"]))
    
    def update_parameters(self, global_param, word_concept_param):
        self.param_general = global_param
        self.param_word_concept = word_concept_param

        if self.scenario_handling == "tiled":
            self.dirmult_obj = DirMultStore(self.param_general["dirichlet_alpha"], self.param_general["num_scenarios"])
        else:
            self.scen_sampling_obj = ScenarioSampler(self.param_scenario_concept, self.param_word_concept, 
                                                     self.param_general["dirichlet_alpha"], self.param_general["num_scenarios"], self.param_general["num_words"],
                                                     int(self.scenario_config["NumSamples"]), int(self.scenario_config["Restarts"]), int(self.scenario_config["Discard"]))
        

class PossiblePolysemy:
    def __init__(self, do_polysemy, vgpath_obj, vgindex_obj, vgobjects_attr_rel, sds_obj):
        self.do_polysemy = do_polysemy

        param_scenario_concept = dict((int(c), slist) for c, slist in sds_obj.param_scenario_concept.items())
        self.poly_obj = SyntheticPolysemes(vgpath_obj, vgindex_obj, vgobjects_attr_rel, param_scenario_concept) if self.do_polysemy else None
        self.next_wordid = len(vgobjects_attr_rel[VGOBJECTS]) + len(vgobjects_attr_rel[VGATTRIBUTES]) + len(vgobjects_attr_rel[VGRELATIONS])

        self.global_param = sds_obj.param_general.copy()
        self.word_concept_param = sds_obj.param_word_concept.copy()

    def transform(self, paragraph, sds_obj, vgindex_obj):
        if self.do_polysemy:
            # transform the paragraph
            paragraph_sep = [ [sid, [ell for ell in sent if ell[0].endswith("w")], [ell for ell in sent if ell[0].endswith("r")]] for sid, sent in paragraph]
            
            paragraph_transformed, goldwords = self.poly_obj.make(paragraph_sep, self.next_wordid, simlevel = 0)
            
            # transformed word literals and roles are separated out; remove that separation
            paragraph_transformed = [[sid, w1+w2+r] for sid, w1, w2, r in paragraph_transformed]

            # adapt parameters for new words
            num_clozewords = len(goldwords)
            global_param_transformed = self.global_param.copy()
            global_param_transformed["num_words"] += num_clozewords

            word_concept_param_transformed = self.word_concept_param.copy()
            # # add to word-concept log probabilities:
            # # cloze word IDs -> concept ID -> logprob
            for word_id, entry in goldwords.items():

                # word cloze ID -> concept ID -> log of 0.5:
                # equal output probability for both concepts
                word_concept_param_transformed[ str(word_id) ] = dict( (str(entry["concept_ids"][i]),  -0.69) for i in [0,1])

            # in the SDS object, add the cloze words to global parameters
            sds_obj.update_parameters(global_param_transformed, word_concept_param_transformed)

            # in the VG index object, store cloze words as additional dictionary
            vgindex_obj.additional_dict = dict( (word_id, (e["word"], e["ctype"])) for word_id, e in goldwords.items())
            
            return (paragraph_transformed, sds_obj, vgindex_obj)
        else:
            return (paragraph, sds_obj, vgindex_obj)

def mentalfile_s(entry, vgindex_obj):
    return "dref " + str(entry["dref"]) + " " + ", ".join([vgindex_obj.ix2l(cix)[0] for cix, val in entry["entry"] if val > 0])

#####
# main loop

# make SDS object
sds_obj = MySDSD(vgpath_obj, config["Scenarios"])

paragraphs = [
    ( "a black cat. the cat.", 
          [
            [0, [["w", 577,0], ["w", 3769, 1], ["r", "arg1", 1, 0]]],
            [1, [["prew", 577, 0]]]
        ]
    ),
    ( "a black cat. the black cat.", 
          [
            [0, [["w", 577,0], ["w", 3769, 1], ["r", "arg1", 1, 0]]],
            [1, [["prew", 577, 0], ["prew", 3769, 1], ["prer", "arg1", 1, 0]]]
        ]
    ),
    ( "a black cat and a cat. the black cat.", 
          [
            [0, [["w", 577,0], ["w", 3769, 1], ["r", "arg1", 1, 0], ["w", 577, 2]]],
            [1, [["prew", 577, 0], ["prew", 3769, 1], ["prer", "arg1", 1, 0]]]
        ]
    ),
    ( "a black cat and a white cat. the black cat.", 
          [
            [0, [["w", 577,0], ["w", 3769, 1], ["r", "arg1", 1, 0], ["w", 577, 2], ["w", 5416, 3], ["r", "arg1", 3,2]]],
            [1, [["prew", 577, 0], ["prew", 3769, 1], ["prer", "arg1", 1, 0]]]
        ]
    ),
    (" a black cat and a black car. the black cat.",
         [
            [0, [["w", 577,0], ["w", 3769, 1], ["r", "arg1", 1, 0], ["w", 550, 2], ["w", 3769, 3], ["r", "arg1", 3, 2]]],
            [1, [["prew", 577, 0], ["prew", 3769, 1], ["prer", "arg1", 1, 0]]]
        ]
    ),      
    ( "a cat on a roof. the cat.", 
          [
            [0, [["w", 577,0], ["w", 2401, 1], ["w", 5501, 2], ["r", "arg0", 2, 0], ["r", "arg1", 2, 1]]],
            [1, [["prew", 577, 0]]]
        ]
    ),
    ( "a cat on a roof. the cat on the roof.", 
          [
            [0, [["w", 577,0], ["w", 2401, 1], ["w", 5501, 2], ["r", "arg0", 2, 0], ["r", "arg1", 2, 1]]],
            [1, [["prew", 577,0], ["prew", 2401, 1], ["prew", 5501, 2], ["prer", "arg0", 2, 0], ["prer", "arg1", 2, 1]]]
        ]
    ),
    ( "a cat on a roof and a cat. the cat on the roof.", 
          [
            [0, [["w", 577,0], ["w", 2401, 1], ["w", 5501, 2], ["r", "arg0", 2, 0], ["r", "arg1", 2, 1], ["w", 577, 3]]],
            [1, [["prew", 577,0], ["prew", 2401, 1], ["prew", 5501, 2], ["prer", "arg0", 2, 0], ["prer", "arg1", 2, 1]]]
        ]
    ),
    ( "a white cat on a roof. the white cat on the roof.", 
          [
            [0, [["w", 577,0], ["w", 2401, 1], ["w", 5501, 2], ["r", "arg0", 2, 0], ["r", "arg1", 2, 1], ["w", 5416, 3], ["r", "arg1", 3, 0]]],
            [1, [["prew", 577,0], ["prew", 2401, 1], ["prew", 5501, 2], ["prer", "arg0", 2, 0], ["prer", "arg1", 2, 1], ["prew", 5416, 3], ["prer", "arg1", 3, 0]]]
        ]
    ),    
    ( "a white cat on a roof and a black cat on a roof. the black cat on the roof.", 
          [
            [0, [["w", 577,0], ["w", 2401, 1], ["w", 5501, 2], ["r", "arg0", 2, 0], ["r", "arg1", 2, 1], ["w", 5416, 3], ["r", "arg1", 3, 0],
                     ["w", 577,4], ["w", 2401, 5], ["w", 5501, 6], ["r", "arg0", 6, 4], ["r", "arg1", 6, 5], ["w", 3769, 7], ["r", "arg1", 7, 4]]],
            [1, [["prew", 577,0], ["prew", 2401, 1], ["prew", 5501, 2], ["prer", "arg0", 2, 0], ["prer", "arg1", 2, 1], ["prew", 3769, 3], ["prer", "arg1", 3, 0]]]
        ]
    ),
    ( "a white cat on a roof. the black cat on the roof. (SHOULD FAIL)", 
          [
            [0, [["w", 577,0], ["w", 2401, 1], ["w", 5501, 2], ["r", "arg0", 2, 0], ["r", "arg1", 2, 1], ["w", 5416, 3], ["r", "arg1", 3, 0]]],
            [1, [["prew", 577,0], ["prew", 2401, 1], ["prew", 5501, 2], ["prer", "arg0", 2, 0], ["prer", "arg1", 2, 1], ["prew", 3769, 3], ["prer", "arg1", 3, 0]]]
        ]
    )   
]

for use_polysemy in [False, True]:
    print("******************************")
    print("Polysemy:", use_polysemy)

    poly_obj = PossiblePolysemy(use_polysemy, vgpath_obj, vgindex_obj, vgobjects_attr_rel, sds_obj)
    
    for description, paragraph in paragraphs:
        # paragraph= poly_obj.transform(paragraph)

        if use_polysemy:
            paragraph, sds_obj, vgindex_obj = poly_obj.transform(paragraph, sds_obj, vgindex_obj)
            
        print("\n\n")
        print("-----Paragraph:", description, "---------")
        ##
        # show paragraph
        for sentence_id, sentence in paragraph:
            print("--Sentence--", sentence_id)
            out_obj.write_sentence(sentence)

        ##
        # compute

        mentalfiles = {"entity": [ ], "roles":[]}
        paragraph_okay = True

        for sentence_id, sentence in paragraph:
            fg = sds_obj.build_factor_graph(sentence, mentalfiles)
            if fg is None:
                # factor graph not successfuly built
                print("Error in creating factor graph for sentence, skipping paragraph.")
                paragraph_okay = False
                break

            # do the inference
            thismap = fg.map_inference()

            print("MAP result", thismap)

            # store new entries to the mental files
            mentalfiles =  sds_obj.extend_mentalfiles_map(thismap, sentence_id, sentence, mentalfiles) 

            if "drefindex" in thismap:
                dref_mfindex = sds_obj.coref_readoff_map(thismap)
                print("Sentence", sentence_id, "coref:")
                for dref, mfindex in dref_mfindex.items():
                    entry = mentalfiles["entity"][mfindex]
                    print("\t", dref, "->", mentalfile_s(entry, vgindex_obj))


        if paragraph_okay:
            print("--")
            print("Mental files: entities")
            for e in mentalfiles['entity']:
                print("\t", mentalfile_s(e, vgindex_obj))
            print("Mental files: roles")
            print(mentalfiles['roles'])
        else:
            print("Failure to process paragraph.")
