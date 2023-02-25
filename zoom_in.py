# Katrin Erk January 2023
# analyze SDS output: zoom in on individual sentences


import sys
import os
import json
import zipfile
from collections import defaultdict, Counter
import math
import numpy as np
import random
import statistics
from argparse import ArgumentParser

import vgiterator
from vgindex import VgitemIndex
from vgpaths import VGPaths
import sentence_util

from sds import SDS

# to fix:
# - option to include cloze words, don't always look for them
# - let user add literals

class InspectSentences:
    def __init__(self, vgindex_obj, sds_obj, topic_obj, out_obj):
        self.vgindex_obj = vgindex_obj
        self.sds_obj = sds_obj
        self.topic_obj = topic_obj
        self.output_obj = out_obj

    ###
    # choose between sentences
    def main_loop(self, sentid_sentence):
        while True:
            user_input = input("""Type sentence ID to inspect sentence,
* to inspect a random sentence,
or q to quit: """)

            if user_input == "q":
                # end
                break

            elif user_input == "*":
                sentid = random.choice(list(sentid_sentence.keys()))
                self.inspect_sentence(sentid, sentid_sentence[sentid])

            elif user_input.isdigit():
                if user_input in sentid_sentence:
                    self.inspect_sentence(user_input, sentid_sentence[user_input])
                else:
                    print("unknown sentence ID")

            print()


                     
    ###
    # inspect a sentence:
    # run MAP inference, print sentence with MAP assignments,
    # then let user choose focus literal or add literals or end
    def inspect_sentence(self, sentence_id, sentence):
        print("running MAP inference...")
        
        # get factor graph for this sentence
        fg = self.sds_obj.build_factor_graph(sentence)

        # obtain overall MAP assignment for this sentence
        mapresult = fg.map_inference()

        # display:
        # - the sentence itself, with a number on each literal, and the MAP
        #   scenario and concept for the literal
        # - characterizations of the 3 main scenarios
        print("Sentence", sentence_id)
        self.output_obj.write_sentence(sentence, scenario_concept_assignment = mapresult)

        # scenarios used in the sentence, counted
        print("\nMain scenarios:")
        scenario_counter = Counter(mapresult["scenario"])
        for scenario, _ in scenario_counter.most_common(3):
            print(scenario, ":", self.topic_obj.topic_info( scenario))

        print()

        while True:
            user_input = input("""Type literal index to choose a focus literal,
or q to stop analyzing this sentence: """)

            if user_input == "q":
                break
            elif user_input.isdigit():
                self.inspect_focus_literal(int(user_input), sentence_id, sentence, fg, mapresult)

            # HIER also let the user add literals to the sentence, rerun MAP

    ########3
    # inspect a focus literal in a sentence:
    # - show marginal probabilities of concepts and scenarios for the focus literal.
    #   also characterizations of top k scenarios for the focus literal
    # - show potential for MAP reading, for the most likely reading with each of the
    #    possible concepts for the focus literal. 
    def inspect_focus_literal(self, focusliteral_index, sentence_id, sentence, fg, mapresult):
        wordliterals = [ell for ell in sentence if ell[0] == "w"]
        focusliteral = wordliterals[focusliteral_index]
        _, focusword, focusdref = focusliteral

        print("\nFocus literal:", self.vgindex_obj.ix2l(focusword)[0] + "(" + str(focusdref) + ")\n")
        
        print("Marginal probabilities:\n")
        marginals = fg.marginal_inference()
        # concept marginals
        print("Focus literal concept marginals:")
        mconc = marginals["concept"][focusliteral_index]
        for conceptid_str, prob in mconc.items():
             print("\t", self.vgindex_obj.ix2l(int(conceptid_str))[0], ":", round(prob, 3))
        print("\nFocus literal scenario marginals:")
        mscen = marginals["scenario"][focusliteral_index]
        for scenarioid_str, prob in mscen.items():
             print("\t", scenarioid_str, ":", round(prob, 3))
        print()

        print("Relevant scenarios:")
        for scenarioid_str in mscen.keys():
            print(scenarioid_str, ":", self.topic_obj.topic_info( int(scenarioid_str)))

        print()
        print("Most likely valuations for each possible concept for the focus literal:")
        # MAP goes first
        logpotential = fg.readoff_valuation(mapresult)
        potential = 0 if logpotential is None else math.exp(logpotential)
            
        mapconcept = mapresult["concept"][focusliteral_index]
        print("Conc.", self.vgindex_obj.ix2l(mapconcept)[0],
              "Scen", mapresult["scenario"][focusliteral_index],
              "Potential", potential)

        # now all others, sorted by probability, highest first
        for conceptid_str, prob in sorted(mconc.items(), key  = lambda p:p[1], reverse = True):
            this_conceptid = int(conceptid_str)
            if this_conceptid == mapconcept:
                # we've already printed the potential for the MAP
                continue

            # make a new factor graph where we force the concept to be this
            this_fg = fg.fg_with_evidence([("concept", focusliteral_index, this_conceptid)])
            # and determine the most likely overall valuation
            # when we force this one concept
            this_map = this_fg.map_inference()

            # then compute the potential of that valuation on the *main*, unaltered factor graph
            thislogp = fg.readoff_valuation(this_map)
            thispotential = 0 if thislogp is None else math.exp(thislogp)
                
            print("Conc.", self.vgindex_obj.ix2l(this_conceptid)[0],
                  "Scen.", this_map["scenario"][focusliteral_index],
                  "Potential", thispotential)
                

        print()
        

#
# do we also want to do:
# - scenario preferences for each concept for the literal
# - selectional preferences for any adjacent predicate
#

    
#################
# main starts here
#############3
# initialization

parser = ArgumentParser()
parser.add_argument('input', help="directory with input sentences")
parser.add_argument('--tilesize', help="Scenario tiling: number of scenarios restricted by one Dir-Mult factor, default: 6", type = int, default = 6)
parser.add_argument('--tileovl', help="Scenario tiling: overlap between tiles, default: 2", type = int, default = 2)
parser.add_argument('--cloze', help="Look for additional cloze-generated word labels. default: False", action = "store_true")


args = parser.parse_args()

vgpath_obj = VGPaths(sdsdata = args.input)

vgcounts_zipfilename, vgcounts_filename = vgpath_obj.vg_counts_zip_and_filename()
with zipfile.ZipFile(vgcounts_zipfilename) as azip:
    with azip.open(vgcounts_filename) as f:
        vgobjects_attr_rel = json.load(f)

########3
# Initialization:
# make wrapper object
# for creating factor graphs
print("initializing...")

sds_obj = SDS(vgpath_obj, tilesize = arg.tilesize, tileovl = arg.tileovl)

# read sentences, store under sentence ID
sentid_sentence = { }
for sentence_id, sentence in sds_obj.each_sentence_json():
    sentid_sentence[ sentence_id ] = sentence

# read cloze wordlabels?
if args.cloze:
    zipfilename, filename = vgpath_obj.sds_gold()
    with zipfile.ZipFile(zipfilename) as azip:
        with azip.open(filename) as f:
            golddata = json.load(f)

    if "cloze" not in golddata:
        # whoops, wrong gold data?
        raise Exception("Was expecting cloze info in gold data", zipfilename)

    # dictionary of cloze words
    cloze_dict = dict( (int(wordid_s), golddata["cloze"]["words"][wordid_s]["word"]) for wordid_s in golddata["cloze"]["words"].keys())
else:
    cloze_dict = None

# mapping between labels and label indices
vgindex_obj = VgitemIndex(vgobjects_attr_rel, additional_index_obj_dict = cloze_dict)

# obtain topic characterizations
topic_obj = sentence_util.TopicInfoUtil(vgpath_obj)

#####
# Main loop:
# user chooses sentence to analyze
print()

out_obj = sentence_util.SentencePrinter(vgindex_obj, with_wordliteral_index = True, show_attrel_unary = True)


main_obj = InspectSentences(vgindex_obj, sds_obj, topic_obj, out_obj)
main_obj.main_loop(sentid_sentence)
