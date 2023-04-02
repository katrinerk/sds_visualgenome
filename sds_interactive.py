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
import configparser

import vgiterator
from vgindex import VgitemIndex
from vgpaths import VGPaths
import sentence_util
from vec_util import VectorInterface
from sds_imagine_util import  ImagineScen, ImagineAttr


from sds import SDS

# to fix:
# - option to include cloze words, don't always look for them
# - let user add literals

class InspectSentences:
    def __init__(self, vgindex_obj, sds_obj, topic_obj, scen_obj, attr_obj, out_obj):
        self.vgindex_obj = vgindex_obj
        self.sds_obj = sds_obj
        self.topic_obj = topic_obj
        self.output_obj = out_obj
        self.scen_obj = scen_obj
        self.attr_obj = attr_obj

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
            user_input = input("""Options:
d to zoom in on disambiguation of a literal,
a to imagine additional attributes for a referent based on a literal, 
o to imagine additional objects in the scene based on the MAP scenario assignment,
q stop analyzing this sentence: 
""")

            if user_input == "q":
                break
            
            elif user_input == "o":
                self.imagine_scen(sentence_id, sentence, mapresult)

            elif user_input == "a":
                literal_id = input("Please choose a literal index: ")
                self.imagine_attr(int(literal_id), sentence_id, sentence, mapresult)
                
            elif user_input == "d":
                literal_id = input("Please choose a literal index: ")
                self.inspect_disambiguation(int(literal_id), sentence_id, sentence, fg, mapresult)



    ########3
    # inspect a focus literal in a sentence:
    # - show marginal probabilities of concepts and scenarios for the focus literal.
    #   also characterizations of top k scenarios for the focus literal
    # - show potential for MAP reading, for the most likely reading with each of the
    #    possible concepts for the focus literal. 
    def inspect_disambiguation(self, focusliteral_index, sentence_id, sentence, fg, mapresult):
        wordliterals = [ell for ell in sentence if ell[0] == "w"]
        focusliteral = wordliterals[focusliteral_index]
        _, focusword, focusdref = focusliteral

        print("\nAnalyzing focus literal:", self.vgindex_obj.ix2l(focusword)[0] + "(" + str(focusdref) + ")\n")
        
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

    def imagine_scen(self, sentence_id, sentence, mapresult, n = 10):
        print("\nImagining additional objects based on scenarios in sentence.", sentence_id)

        scenario_list = mapresult["scenario"]
        model_objectid_ranking, _ = self.scen_obj.predict_objectids(scenario_list)
        print("Top", n, ":")
        print(", ".join(self.vgindex_obj.ix2l(oid)[0] for oid in model_objectid_ranking[:n]))
        print()
        

    def imagine_attr(self, focusliteral_index, sentence_id, sentence, mapresult):
        # confirming literal
        wordliterals = [ell for ell in sentence if ell[0] == "w"]
        focusliteral = wordliterals[focusliteral_index]
        _, focuswordix, focusdref = focusliteral

        # obtaining concept
        conceptix = mapresult["concept"][focusliteral_index]
        focusconcept = self.vgindex_obj.ix2l(conceptix)[0]
        
        print("\nPredicting attributes for focus literal:", self.vgindex_obj.ix2l(focuswordix)[0] + "(" + str(focusdref) + ") Concept", focusconcept, "\n")


        Ypredict, _, used_obj_labels = self.attr_obj.predict_forobj([ focusconcept])
        if len(used_obj_labels) != 1:
            print("Could not predict any attributes.")
            return

        Ypredict = Ypredict.tolist()        
        attlist = self.attr_obj.attributelabels
        
        top_attributes = [a for a, _ in sorted(zip(attlist, Ypredict[0]), key = lambda p:p[1], reverse = True)][:10]

        
        print("\nTop predicted attributes:")
        print(", ".join( top_attributes))
        
        print()
        

    
#################
# main starts here
#############3
# initialization

parser = ArgumentParser()
parser.add_argument('input', help="directory with input sentences")
parser.add_argument('--cloze', help="Look for additional cloze-generated word labels. default: False", action = "store_true")
parser.add_argument('--num_att', help = "number of top attributes to use for prediction, default 500", type = int, default = 500)
parser.add_argument('--plsr_components', help = "number of components to use for PLSR, default 100", type = int, default = 100)


args = parser.parse_args()

# settings file
config = configparser.ConfigParser()
config.read("settings.txt")

vgpath_obj = VGPaths(sdsdata = args.input)

vgcounts_zipfilename, vgcounts_filename = vgpath_obj.vg_counts_zip_and_filename()
with zipfile.ZipFile(vgcounts_zipfilename) as azip:
    with azip.open(vgcounts_filename) as f:
        vgobjects_attr_rel = json.load(f)

########3
# Initialization:
# make wrapper object
# for creating factor graphs
print("reading data...")

sds_obj = SDS(vgpath_obj, config["Scenarios"])

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

# imagining objects from scenarios
scen_obj = ImagineScen(vgpath_obj, vgindex_obj)

# imagine attributes from objects
print("learning attribute predictions...")
vgiter = vgiterator.VGIterator(vgobjects_attr_rel)
vec_obj = VectorInterface(vgpath_obj)

attr_obj = ImagineAttr(vgiter, vec_obj, num_attributes = args.num_att, num_plsr_components = args.plsr_components)
# print("HIER I have attributes for", attr_obj.used_training_objectlabels)

#####
# Main loop:
# user chooses sentence to analyze
print()

out_obj = sentence_util.SentencePrinter(vgindex_obj, with_wordliteral_index = True, show_attrel_unary = True)

main_obj = InspectSentences(vgindex_obj, sds_obj, topic_obj, scen_obj, attr_obj, out_obj)
main_obj.main_loop(sentid_sentence)
