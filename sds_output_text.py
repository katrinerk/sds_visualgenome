# Katrin Erk March 2023
# given SDS output, turn into text


import sys
import os
import json
import zipfile
from collections import defaultdict, Counter
import numpy as np
from argparse import ArgumentParser

import vgiterator
import sentence_util
from vgindex import VgitemIndex
from vgpaths import VGPaths, get_output_path

########
parser = ArgumentParser()
parser.add_argument('sdsinput', help="directory with input to SDS")
parser.add_argument('sdsoutput', help="directory with output from SDS")
parser.add_argument('--cloze', help="Look for additional cloze-generated word labels. default: False", action = "store_true")
parser.add_argument('--allunary', help="Show all unary literals, not just those denoting objects. default: False", action = "store_true")

args = parser.parse_args()

####
# read data

vgpath_obj = VGPaths(sdsdata = args.sdsinput, sdsout =  args.sdsoutput)

vgcounts_zipfilename, vgcounts_filename = vgpath_obj.vg_counts_zip_and_filename()
with zipfile.ZipFile(vgcounts_zipfilename) as azip:
    with azip.open(vgcounts_filename) as f:
        vgobjects_attr_rel = json.load(f)



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
    cloze_dict = dict( (int(wordid_s), (golddata["cloze"]["words"][wordid_s]["word"], golddata["cloze"]["words"][wordid_s].get("ctype", "obj"))) \
                            for wordid_s in golddata["cloze"]["words"].keys())
else:
    cloze_dict = None


# mapping between labels and label indices
vgindex_obj = VgitemIndex(vgobjects_attr_rel, additional_dict = cloze_dict)

# obtain topic characterizations
topic_obj = sentence_util.TopicInfoUtil(vgpath_obj)

out_obj = sentence_util.SentencePrinter(vgindex_obj, with_wordliteral_index = False, show_attrel_unary = args.allunary)

# read sentences
zipfilename, filename = vgpath_obj.sds_sentence_zipfilename()
with zipfile.ZipFile(zipfilename) as azip:
    with azip.open(filename) as f:
        sentences = json.load(f)

# and store as mapping from sentence ID to sentence
sentid_sent = dict( (e["sentence_id"], e["sentence"]) for e in sentences)

# read MAP results from SDS
zipfilename, filename = vgpath_obj.sds_output_zipfilename()
with zipfile.ZipFile(zipfilename) as azip:
    with azip.open(filename) as f:
        model_data = json.load(f)

for thismodel in model_data:
    # retrieve sentence
    sentence_id = thismodel["sentence_id"]
    if sentence_id not in sentid_sent:
        raise Exception("Sentence not found " + sentence_id)

    sentence = sentid_sent[sentence_id]
    print("====")
    print("Sentence", sentence_id)
    out_obj.write_sentence(sentence, scenario_concept_assignment = thismodel["MAP"])
    print("--")
    print("\nMain scenarios:")
    scenario_counter = Counter(thismodel["MAP"]["scenario"])
    for scenario, _ in scenario_counter.most_common(3):
        print(scenario, ":", topic_obj.topic_info( scenario))

    print()
    

    
