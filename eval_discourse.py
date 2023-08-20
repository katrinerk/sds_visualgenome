# Katrin Erk June 2023
# evaluate SDS output on discourse and coreference task


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
parser.add_argument('--sdsinput', help="directory with input to sds, default: sds_in/discourse", default = "sds_in/discourse/")
parser.add_argument('--sdsoutput', help="directory with sds output, default: sds_out/discourse", default = "sds_out/discourse/")
parser.add_argument('--outdir', help="directory to write output for inspection, default: inspect_output/discourse", default = "inspect_output/discourse/")
parser.add_argument('--vgdata', help="directory with VG data including frequent items, train/test split, topic model", default = "data/")

args = parser.parse_args()


########3

vgpath_obj = VGPaths(sdsdata = args.sdsinput, sdsout =  args.sdsoutput, vgdata = args.vgdata)

# read frequent object counts
vgcounts_zipfilename, vgcounts_filename = vgpath_obj.vg_counts_zip_and_filename()
with zipfile.ZipFile(vgcounts_zipfilename) as azip:
    with azip.open(vgcounts_filename) as f:
        vgobjects_attr_rel = json.load(f)

# read gold data
zipfilename, filename = vgpath_obj.sds_gold()
with zipfile.ZipFile(zipfilename) as azip:
    with azip.open(filename) as f:
        gold_data = json.load(f)

if "discourse" not in gold_data:
    # whoops, wrong gold data?
    raise Exception("Doing discourse evaluation, was expecting discourse info in gold data", zipfilename)

if gold_data["discourse"]["cond"] in ["poly", "tpoly"]:
    # synthetic polysemy, need to have cloze data
    if "cloze" not in gold_data:
        raise Exception("Synthetic polysemy in discourse data: need cloze entries in gold data structure")

    # make dictionary of cloze words for vgindex object
    # dictionary of cloze words
    cloze_dict = dict( (int(wordid_s), (gold_data["cloze"][wordid_s]["word"], gold_data["cloze"][wordid_s].get("ctype", VGOBJECTS))) \
                            for wordid_s in gold_data["cloze"].keys())

else:
    cloze_dict = None
    
# read passages
zipfilename, filename = vgpath_obj.sds_sentence_zipfilename()
with zipfile.ZipFile(zipfilename) as azip:
    with azip.open(filename) as f:
        passages = json.load(f)

# read MAP results from SDS
zipfilename, filename = vgpath_obj.sds_output_zipfilename()
with zipfile.ZipFile(zipfilename) as azip:
    with azip.open(filename) as f:
        model_data = json.load(f)

vgindex_obj = VgitemIndex(vgobjects_attr_rel, additional_dict = cloze_dict)
    
        

#################################
# evaluation: compute correctness of reference for the one definite description in each passage
# also store which combinations of sentente IDs we've processed,
# so we can deduct points for passages that could not be processed.

def model_sentids_for_comparegold(model_dict, condition):
    if condition == "tpoly":
        # remove the last sentence ID, it is a duplicate of the penultimate one.
        # but only for comparison with gold sentence ID lists.
        # the actual passage data will have the duplicates
        
        return model_dict["sentids"][:-1]
    else:
        return model_dict["sentids"]

## iterator over passages with model and gold labels
def each_passage_model_and_gold(model_data, gold_data, sentence_data, condition):
    
    for model_dict in model_data:
        gold_found = False
        passage_found = False

        # look for gold entry to match model entry
        for gold_dict in gold_data["discourse"]["passages"]:
            if gold_dict["sentids"] == model_sentids_for_comparegold(model_dict, condition):
                # found
                gold_found = True

                # look for passage to match model entry
                for passage in sentence_data:
                    passage_sentids = [s["sentence_id"] for s in passage]
                    if passage_sentids == model_dict["sentids"]:
                        # match
                        passage_found = True
                        yield (model_dict, gold_dict, passage)

        # should not be here: no passage found to match model data
        if not passage_found:
            raise Exception("no passage found matching model data " + " ".join(model_dict["sentids"]))

        # should not be here: no gold data found to match model data
        if not gold_found: 
            raise Exception("no gold data found matching model data " + " ".join(model_dict["sentids"]))
    

# store each set of sentence IDs processed, so that we can later check
# if any passages were skipped because of processing errors
sentids_processed = set()
# for each passage: 1 = correct coref, 0 = incorrect.
# separately for each chain length, and "all"
correct_coref = defaultdict(list)

# for all passages that were processed,
# check whether coreference was guessed correctly
for model_dict, gold_dict, passage in each_passage_model_and_gold(model_data, gold_data, passages, gold_data["discourse"]["cond"]):
    # remember that we processed this passage, so we can compare to gold passages later
    sentids_processed.add( tuple(model_sentids_for_comparegold(model_dict, gold_data["discourse"]["cond"])) )

    # find the coreferences computed for this passage
    if len(model_dict["coref"]) == 0:
        # no coreferences recorded: this is an error, there is
        # always exactly one definite description in each passage
        print("no coreference found in passage, counting as wrong")
        correct_coref["all"].append(0)
        correct_coref[str(gold_dict["chain"])].append(0)
        continue

    # gold: sentence ID, discourse referent
    target_pairs = gold_dict["target"]

    # check whether each gold mapping matches some model mapping.
    thiscorrect = 1
    for from_sentid_dref, to_sentid_dref in target_pairs:
        if not(any(f == from_sentid_dref and t == to_sentid_dref for f, t, concepts in model_dict["coref"])):
            # no match found
            thiscorrect = 0
            break
    
    correct_coref["all"].append(thiscorrect)
    correct_coref[str(gold_dict["chain"])].append(thiscorrect)
    
# check if we missed any passages
missed_passages = { }
missed_passages["all"] = [gold_dict for gold_dict in gold_data["discourse"]["passages"] if tuple(gold_dict["sentids"]) not in sentids_processed]
correct_coref["all"] += len(missed_passages["all"]) * [0]

for chainlength in list(correct_coref.keys()):
    if chainlength == "all": continue
    missed_passages[str(chainlength)] =  [gold_dict for gold_dict in gold_data["discourse"]["passages"] \
                                    if gold_dict["chain"] == chainlength and tuple(gold_dict["sentids"]) not in sentids_processed]
    
    correct_coref[str(chainlength)] += len(missed_passages[str(chainlength)]) * [0]



# report performance
print("--------- Coreference evaluation -----")
for key, label in [ ("all", "Overall")] + [(chainlength, "Chainlength " + str(chainlength)) for chainlength in sorted(correct_coref.keys()) if chainlength != "all"]:
    print(label, "evaluation:")
    
    print("\tCorrect:", round(sum(correct_coref[key]) / len(correct_coref[key]), 3), "which is", sum(correct_coref[key]), "out of", len(correct_coref[key]))
    if sum(correct_coref[key]) < len(correct_coref[key]):
        print("\tOf the incorrect cases, number of missing passages:", len(missed_passages[key]))


# Condition tpoly gets an extra evaluation:
# is disambiguation more often correct in the probe sentences, which
# should be disambiguated by their antecedent,
# than in their non-probe counterparts?
if gold_data["discourse"]["cond"] == "tpoly":
    print("\n------- Disambiguation evaluation --------")
    
    # disamb_correct: a list of 0s and 1s, separately for probe and comparison sentence
    disamb_correct = defaultdict(list)
    
    for model_dict, gold_dict, passage in each_passage_model_and_gold(model_data, gold_data, passages, gold_data["discourse"]["cond"]):
        # probe sentence and comparison sentence have the same content, and the same polysemy. 
        # The difference is that the probe sentence is a definite description.
        # so the probe sentence should be disambiguated by its antecedent, but the comparison sentence
        # only has selectional constraints and scenario to do the disambiguation
        probe_sentobj = passage[-2]
        comparison_sentobj = passage[-1]

        # evaluate polysemy in both sentences
        for label, sentobj, sentindex in [ ("probe", probe_sentobj, -2), ("comparison", comparison_sentobj, -1) ]:
            words =  [ell for ell in sentobj["sentence"] if ell[0].endswith("w")]

            for wordindex, word in enumerate(words):
                _, word_id, _ = word
                if str(word_id) in gold_data["cloze"]:
                    # this is a word to be disambiguated
                    modelconcept = model_dict["MAP"][sentindex]["concept"][wordindex]
                    goldconcept = gold_data["cloze"][str(word_id)]["gold_id"]

                    disamb_correct[label].append( int( modelconcept == goldconcept) )


    for label in ["probe", "comparison"]:
        print(label, "sentences, accuracy of disambiguation:", round(sum(disamb_correct[label]) / len(disamb_correct[label]), 3))

        
# and write some randomly selected passages out for inspection

random.seed(100)
num_passages = len(correct_coref)
passage_indices_for_inspection = random.sample(list(range(num_passages)), min(100, num_passages))

outpath = get_output_path(os.path.join(args.outdir, "eval_discourse_out.txt"))
with open(outpath, "w") as outf:
    out_obj = sentence_util.SentencePrinter(vgindex_obj, file = outf)

    passage_index = 0
    for model_dict, gold_dict, passage in each_passage_model_and_gold(model_data, gold_data, passages, gold_data["discourse"]["cond"]):
        if passage_index in passage_indices_for_inspection:
            # write out this passage
            print("\n\n----------------", file = outf)
            for sent_obj in passage:
                print("--Sentence", sent_obj["sentence_id"], file = outf)
                out_obj.write_sentence(sent_obj["sentence"])

            print("\n", file = outf)
            correctness = "correct" if correct_coref[passage_index] == 1 else "incorrect"
            print(f"Coreference is {correctness}", file = outf)
            
            print("gold coref:", gold_dict["target"], file = outf)
            for from_pair, to_pair in gold_dict["target"]:
                print("\tsentid", from_pair[0], "dref", from_pair[1], "=> sentid", to_pair[0], "dref", to_pair[1], file = outf)
                
            print("predicted coref:", file = outf)
            for from_pair, to_pair, _ in model_dict["coref"]:
                print("\tsentid", from_pair[0], "dref", from_pair[1], "=> sentid", to_pair[0], "dref", to_pair[1], file = outf)
            print("\n", file = outf)

        passage_index += 1
                


                
            
                
