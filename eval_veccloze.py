# Katrin Erk May 2023
# evaluate SDS output on vector cloze task


import sys
import os
import json
import zipfile
from collections import defaultdict, Counter
import numpy as np
from argparse import ArgumentParser
import random

from vgnames import VGOBJECTS, VGATTRIBUTES, VGRELATIONS 
import vgiterator
import sentence_util
from vgindex import VgitemIndex
from vgpaths import VGPaths, get_output_path

########

parser = ArgumentParser()
parser.add_argument('--sdsinput', help="directory with input to sds, default: sds_in/veccloze", default = "sds_in/veccloze/")
parser.add_argument('--sdsoutput', help="directory with sds output, default: sds_out/veccloze", default = "sds_out/veccloze/")
parser.add_argument('--outdir', help="directory to write output for inspection, default: inspect_output/cloze", default = "inspect_output/veccloze/")
parser.add_argument('--vgdata', help="directory with VG data including frequent items, train/test split, topic model", default = "data/")

args = parser.parse_args()


########3

random.seed(100)

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
if "baseline" not in golddata:
    print(golddata)
    raise Exception("missing baseline data in", zipfilename)
    
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

# dictionary of cloze words
cloze_dict = dict( (int(wordid_s), (golddata["cloze"]["words"][wordid_s]["word"], golddata["cloze"]["words"][wordid_s].get("ctype", VGOBJECTS))) \
                            for wordid_s in golddata["cloze"]["words"].keys())

vgindex_obj = VgitemIndex(vgobjects_attr_rel, additional_dict = cloze_dict)
    
        

#################################
# evaluation: compute correctness for each occurrence, both for the model and a frequency baseline
# store sequence of 0's and 1ns (incorrect/correct) separately for each cloze word ID

## iterator over sentences with model and gold labels
#
# model_data: list of sentences. for each sentence:
#  dictionary with entries "sentence_id",
#  "MAP": "concept": dict mapping word index to concept ID
# gold_data: "cloze":"words" : word_id : "gold_id" : concept ID
# sentence_data: sentence_id : sentence
#
# yields tuples (sentence_id, list of model and gold labels)
# list of model and gold labels: list of entries
# (word index, word ID, model concept ID, gold concept ID, baseline concept ID)
def each_sentence(model_data, gold_data, sentence_data, vgindex_obj):

    baseline_freq = gold_data["baseline"]
    for thismodel in model_data:
    
        sentence_id = thismodel["sentence_id"]
    
        # retrieve the original sentence
        if sentence_id not in sentence_data:
            raise Exception("Sentence not found " + sentence_id)

        sentence = sentence_data[sentence_id]
        wordliterals = [ell for ell in sentence if ell[0] == "w"]

        # retrieve model, gold, baseline info to go with this sentence
        model_gold_base_entries = [ ]

        for wordindex, word in enumerate(wordliterals):
            _, word_id, _ = word
            if str(word_id) in gold_data["cloze"]["words"]:
                # this is a word to be disambiguated
                modelconcept = thismodel["MAP"]["concept"][wordindex]
                goldconcept = gold_data["cloze"]["words"][str(word_id)]["gold_id"]
                bothconcept_ids = gold_data["cloze"]["words"][str(word_id)]["concept_ids"]
                for cid in bothconcept_ids:
                    if str(cid) not in baseline_freq:
                        print("missing concept", cid, vgindex_obj.ix2l(cid)[0])
                freqs = [baseline_freq.get(str(cid), 0) for cid in bothconcept_ids]
                baseconcept = bothconcept_ids[0] if freqs[0] > freqs[1] else bothconcept_ids[1]

                model_gold_base_entries.append( (wordindex, word_id, modelconcept, goldconcept, baseconcept) )
            
        yield(sentence_id, model_gold_base_entries, thismodel)


# for each data point, record correct = 1, incorrect = 0
model_correct = [ ]
baseline_correct = [ ]

# sample some sentences for human-readable output
sentids_for_inspection = random.sample(list(sentid_sent.keys()), min(100, len(sentid_sent)))
sentences_for_inspection = [ ]

###
# main evaluation loop
for sentence_id, entries, mapresult in each_sentence(model_data, golddata, sentid_sent, vgindex_obj):
    for wordindex, word_id, modelconcept, goldconcept, baseconcept in entries:
        model_correct.append( int(goldconcept == modelconcept) )
        baseline_correct.append( int(goldconcept == baseconcept) )


    if sentence_id in sentids_for_inspection:
        sentences_for_inspection.append((sentence_id, entries, mapresult))

###
# print to screen: summary evaluation results

print("------------")
print("Accuracy:", "Model:", round(sum(model_correct) / len(model_correct), 3), "Baseline:", round(sum(baseline_correct) / len(baseline_correct), 3))

    
    
###
 # write to file: per-clozeword evaluation and list of wrongly classified cases

# obtain topic characterizations
topic_obj = sentence_util.TopicInfoUtil(vgpath_obj)


outpath = get_output_path(os.path.join(args.outdir, "eval_imagine_veccloze_out.txt"))
with open(outpath, "w") as outf:
    out_obj = sentence_util.SentencePrinter(vgindex_obj, file = outf)

    for sentid, clozedata, mapresult in sentences_for_inspection:
        print("--------Sentence", sentid, "----------", file = outf)
        for wordindex, word_id, modelconcept, goldconcept, baseconcept in clozedata:
            accword = "correct" if goldconcept == modelconcept else "incorr."
            print("Word no.", wordindex, accword, "gold", "'" + "/".join(vgindex_obj.ix2l(goldconcept)) + "'", "assigned", "'" + "/".join(vgindex_obj.ix2l(modelconcept)) + "'", file = outf)

        print(file = outf)
        out_obj.write_sentence(sentid_sent[sentid])
        print(file = outf)
        print("\nMain scenarios:", file = outf)
        scenario_counter = Counter(mapresult["MAP"]["scenario"])
        for scenario, _ in scenario_counter.most_common(3):
            print(scenario, ":", topic_obj.topic_info( scenario), file = outf)

        print(file = outf)
        

                        
                


                
            
                
