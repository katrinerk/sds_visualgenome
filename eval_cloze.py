# Katrin Erk January 2023
# evaluate SDS output on cloze task:
# global evaluation, not zoom-in


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
parser.add_argument('--sdsinput', help="directory with input to sds, default: sds_in/cloze", default = "sds_in/cloze/")
parser.add_argument('--sdsoutput', help="directory with sds output, default: sds_out/cloze", default = "sds_out/cloze/")
parser.add_argument('--outdir', help="directory to write output for inspection, default: inspect_output/cloze", default = "inspect_output/cloze/")
parser.add_argument('--vgdata', help="directory with VG data including frequent items, train/test split, topic model", default = "data/")

args = parser.parse_args()


########3

vgpath_obj = VGPaths(sdsdata = args.sdsinput, sdsout =  args.sdsoutput, vgdata = args.vgdata)

# read frequent object counts
vgcounts_zipfilename, vgcounts_filename = vgpath_obj.vg_counts_zip_and_filename()
with zipfile.ZipFile(vgcounts_zipfilename) as azip:
    with azip.open(vgcounts_filename) as f:
        vgobjects_attr_rel = json.load(f)

# read cloze gold data
zipfilename, filename = vgpath_obj.sds_gold()
with zipfile.ZipFile(zipfilename) as azip:
    with azip.open(filename) as f:
        golddata = json.load(f)

if "cloze" not in golddata:
    # whoops, wrong gold data?
    raise Exception("Doing cloze evaluation, was expecting cloze info in gold data", zipfilename)
    
# read sentences
zipfilename, filename = vgpath_obj.sds_sentence_zipfilename()
with zipfile.ZipFile(zipfilename) as azip:
    with azip.open(filename) as f:
        sentences = json.load(f)

# and store as mapping from sentence ID to sentence
sentid_sent = dict( (e["sentence_id"], e["sentence"]) for e in sentences)

# read MAP and marginal inference results from SDS
zipfilename, filename = vgpath_obj.sds_output_zipfilename()
with zipfile.ZipFile(zipfilename) as azip:
    with azip.open(filename) as f:
        model_data = json.load(f)

#################################
# evaluation: compute correctness for each occurrence, both for the model and a frequency baseline
# store sequence of 0's and 1ns (incorrect/correct) separately for each cloze word ID

def each_sentence(model_data, gold_data, sentence_data):

    for thismodel in model_data:
    
        # retrieve gold info to go with this sentence
        sentence_id = thismodel["sentence_id"]
    
        if sentence_id not in gold_data["cloze"]["sent"]:
            raise Exception("Gold data not found for sentence " + sentence_id)
    
        thisgold = golddata["cloze"]["sent"][sentence_id]

        # and retrieve the original sentence
        if sentence_id not in sentence_data:
            raise Exception("Sentence not found " + sentence_id)

        sentence = sentence_data[sentence_id]
        wordliterals = [ell for ell in sentence if ell[0] == "w"]

        # gold data contains correct underlying concept for each cloze word
        for clozeword_id, gold_concept_id in thisgold:
            baseline_concept_id = golddata["cloze"]["words"][str(clozeword_id)]["baseline_id"]

            for wordindex, literal in enumerate(wordliterals):
                if literal[1] == clozeword_id:
                    # this is an occurrence of this cloze word
                    model_concept_id = thismodel["MAP"]["concept"][wordindex]

                    yield(sentence_id, clozeword_id, wordindex, gold_concept_id, model_concept_id, baseline_concept_id)


# for each data point, record correct = 1, incorrect = 0
# overall, and per cloze word ID
model_correct = [ ]
baseline_correct = [ ]
clozeword_model_correct = defaultdict(list)
clozeword_baseline_correct = defaultdict(list)

error_details = defaultdict(list)

for sentence_id, clozeword_id, wordliteral_index, gold_concept_id, model_concept_id, baseline_concept_id in each_sentence(model_data, golddata, sentid_sent):
    model_correct.append( int(gold_concept_id == model_concept_id) )
    baseline_correct.append( int(gold_concept_id == baseline_concept_id) )

    clozeword_model_correct[ clozeword_id ].append( int(gold_concept_id == model_concept_id) )
    clozeword_baseline_correct[ clozeword_id ].append( int(gold_concept_id == baseline_concept_id) )

    if gold_concept_id != model_concept_id:
        # wrongly classified, store details
        error_details[clozeword_id].append( (sentence_id, wordliteral_index, gold_concept_id, model_concept_id) )


###
# print to screen: summary evaluation results

print("------------")
print("Overall accuracy")
print("------------")
print("Micro-average:", "Model:", round(sum(model_correct) / len(model_correct), 3), "Baseline:", round(sum(baseline_correct) / len(baseline_correct), 3))

clozeword_model_acc = dict( (cid, sum(vals) / len(vals)) for cid, vals in clozeword_model_correct.items())
clozeword_baseline_acc = dict( (cid, sum(vals) / len(vals)) for cid, vals in clozeword_baseline_correct.items())

print("Macro-average:", "Model:", round(sum(clozeword_model_acc.values()) / len(clozeword_model_acc.values()), 3),
      "Baseline:", round(sum(clozeword_baseline_acc.values()) / len(clozeword_baseline_acc.values()), 3))

print()
print("------------")
print("Accuracy by cloze word")
print("------------")
for clozeword_id in sorted(clozeword_model_acc.keys(), key = lambda c: sum(golddata["cloze"]["words"][str(c)]["freq"]), reverse = True):
    concepts = golddata["cloze"]["words"][str(clozeword_id)]["labels"]
    clozeword = golddata["cloze"]["words"][str(clozeword_id)]["word"]
    freq = golddata["cloze"]["words"][str(clozeword_id)]["freq"]
    print(clozeword)
    print("\tModel:", round(clozeword_model_acc[clozeword_id], 3), "Baseline:", round(clozeword_baseline_acc[clozeword_id], 3),
          "Training freq", ", ".join([str(f) for f in freq]))
            
    
    
###
 # write to file: per-clozeword evaluation and list of wrongly classified cases

# obtain topic characterizations
topic_obj = sentence_util.TopicInfoUtil(vgpath_obj)

# dictionary of cloze words
cloze_dict = { }
for wordid_s in golddata["cloze"]["words"].keys():
    cloze_dict[ int(wordid_s)] = (golddata["cloze"]["words"][wordid_s]["word"], "obj")

vgindex_obj = VgitemIndex(vgobjects_attr_rel, additional_dict = cloze_dict)
    

for clozeword_id in clozeword_model_acc.keys():
    ######
    # determine cloze as word, remove all spaces that it may include
    clozeword = golddata["cloze"]["words"][str(clozeword_id)]["word"].replace(" ", "_")
    concepts = golddata["cloze"]["words"][str(clozeword_id)]["labels"]
    frequencies = golddata["cloze"]["words"][str(clozeword_id)]["concepts"]

    ########
    # store file under the name of the cloze word
    outpath = get_output_path(os.path.join(args.outdir, clozeword + ".txt"), overwrite_warning = False)

    with open(outpath, "w") as outf:
        out_obj = sentence_util.SentencePrinter(vgindex_obj, file = outf)
        
        print("Cloze word ", clozeword, ". Concepts: ", concepts[0], " (freq ", freq[0], "), ", concepts[1], " (freq ", freq[1], ")", sep='', file = outf)
        print("Accuracy model:", round(clozeword_model_acc[clozeword_id], 3), "baseline:", round(clozeword_baseline_acc[clozeword_id], 3), file = outf)
        print("---------------------\n\n", file = outf)

        for sentence_id, target_index, gold_concept_id, model_concept_id in error_details[clozeword_id]:
            sentence = sentid_sent[ sentence_id]

            print("-------------------", file = outf)
            print("Sentence", sentence_id, "Gold:", vgindex_obj.ix2l(gold_concept_id)[0], "Assigned:", vgindex_obj.ix2l(model_concept_id)[0], file = outf)
            print(file= outf)

            #######
            # obtain model predictions for this sentence
            found = False
            thismodel = None
            for m in model_data:
                if m["sentence_id"] == sentence_id:
                    found = True
                    thismodel = m

            if not found:
                raise Exception("Error, couldn't find model predictions for sentence ID " + str(sentence_id))
            

            ###########
            # writing the sentence itself
            print("Sentence:", file = outf)
            out_obj.write_sentence(sentence, targetindex = target_index)
            
            print(file = outf)

            ##########
            # print MAP scenario assignment for the sentence.
            print("Overall MAP scenario assignment:", ", ".join([str(v) for v in sorted(thismodel["MAP"]["scenario"])]), "\n", file = outf)


            ##
            # print sketch of the assigned scenario for the target
            assigned_topic = thismodel["MAP"]["scenario"][target_index]
            print("Target assigned scenario:", assigned_topic, file = outf)

            topicstr = topic_obj.topic_info( assigned_topic)
            if topicstr is None:
                raise Exception("topic description not found for topic " + str(assigned_topic))
            else:
                print(topicstr, file = outf)
                    
            print(file = outf)
            
                        
                


                
            
                
