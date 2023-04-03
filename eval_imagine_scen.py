# Katrin Erk February 2023
# evaluate SDS on task of
# predicting additional objects from the
# scenarios in a sentence

import sys
import os
import json
import zipfile
from collections import defaultdict, Counter
import math
import numpy as np
import random
from argparse import ArgumentParser

import vgiterator
import sentence_util
from vgindex import VgitemIndex
from vgpaths import VGPaths, get_output_path
from sds_imagine_util import  ImagineScen

########3
# settings


parser = ArgumentParser()
parser.add_argument('--sdsinput', help="directory with input to sds, default: sds_in/imagine_scen", default = "sds_in/imagine_scen/")
parser.add_argument('--sdsoutput', help="directory with sds output, default: sds_out/imagine_scen", default = "sds_out/imagine_scen/")
parser.add_argument('--outdir', help="directory to write output for inspection, default: inspect_output/imagine_scen", default = "inspect_output/imagine_scen/")
parser.add_argument('--vgdata', help="directory with VG data including frequent items, train/test split, topic model", default = "data/")
parser.add_argument('--num_inspect', help = "number of sentences to write for inspection, default 10", type = int, default = 10)

args = parser.parse_args()

########3
print("Reading data")
vgpath_obj = VGPaths(sdsdata = args.sdsinput, sdsout =  args.sdsoutput, vgdata = args.vgdata)


# frequent objects, attributes, relations
vgcounts_zipfilename, vgcounts_filename = vgpath_obj.vg_counts_zip_and_filename()
with zipfile.ZipFile(vgcounts_zipfilename) as azip:
    with azip.open(vgcounts_filename) as f:
        vgobjects_attr_rel = json.load(f)

# read gold data
zipfilename, filename = vgpath_obj.sds_gold()
with zipfile.ZipFile(zipfilename) as azip:
    with azip.open(filename) as f:
        golddata = json.load(f)

if "imagine_scen" not in golddata:
    # whoops, wrong gold data?
    raise Exception("Doing scenario enrichment evaluation, was expecting imagine_scen in gold data", zipfilename)
    
# read sentences
zipfilename, filename = vgpath_obj.sds_sentence_zipfilename()
with zipfile.ZipFile(zipfilename) as azip:
    with azip.open(filename) as f:
        sentences = json.load(f)

# and store as mapping from sentence ID to sentence
sentid_sent = dict( (e["sentence_id"], e["sentence"]) for e in sentences)

# read MAP inference results from SDS
zipfilename, filename = vgpath_obj.sds_output_zipfilename()
with zipfile.ZipFile(zipfilename) as azip:
    with azip.open(filename) as f:
        model_data = json.load(f)

###########
# get training data counts for objects, for the
# baseline
print("obtaining baseline object frequencies")

split_zipfilename, split_filename = vgpath_obj.vg_traintest_zip_and_filename()
with zipfile.ZipFile(split_zipfilename) as azip:
    with azip.open(split_filename) as f:
        traintest_split = json.load(f)
    
trainset = set(traintest_split["train"])

vgobj = vgiterator.VGIterator(vgobjects_attr_rel)
vgindex_obj = VgitemIndex(vgobjects_attr_rel)

objname_count = Counter()

for img, obj in vgobj.each_image_objects(img_ids = trainset):
    objname_count.update(obj)

# most_common() without argument returns all key/count pairs, most frequent first.
# replace object names by object IDs.
baseline_objectid_ranking = [vgindex_obj.o2ix(objectname) for objectname, _ in objname_count.most_common()]
baseline_objectid_logprob = np.log(np.array([count for _, count in objname_count.most_common()]) / sum(objname_count.values()))

##########################
# for each scenario, obtain a numpy array
# of object log probabilities. to do this,
# kick out attribute and relation log probabilities and renormalize

print("reading scenario/concept probabilities")

scen_obj = ImagineScen(vgpath_obj, vgindex_obj)


topic_obj = sentence_util.TopicInfoUtil(vgpath_obj)

    
#################################
# evaluation: first do the actual prediction of additional objects based on the MAP scenario assignment,
# then compute mean average precision


# average precision
def averageprecision(predicted, gold):
    precs_at_k =  [ ]
    truepos_so_far = 0

    for k, prediction in enumerate(predicted):
        if prediction in gold:
            truepos_so_far += 1

            precs_at_k.append( truepos_so_far / (k+1))

    return sum(precs_at_k) / truepos_so_far

# highest correct object
def highestcorrect(predicted, gold):
    for ix, prediction in enumerate(predicted):
        if prediction in gold:
            return ix

# perplexity for a series of observations.
# but keep conditional probability conditioned
# only on the seen document not the other unseen observations
def perplexity(objects, predicted_logprob, new_objects):
    return math.exp(- (1 / len(new_objects)) * sum([ logprob for o, logprob in zip(objects, predicted_logprob) if o in new_objects]))
    
        
#############
# iterate over sentences, retrieve gold hidden objects
# and MAP scenario prediction
def each_sentence(model_data, gold_data, sentence_data): 

    for thismodel in model_data:
    
        # retrieve gold info to go with this sentence
        sentence_id = thismodel["sentence_id"]
    
        if sentence_id not in gold_data["imagine_scen"]:
            raise Exception("Gold data not found for sentence " + sentence_id)

        # and retrieve the original sentence
        if sentence_id not in sentence_data:
            raise Exception("Sentence not found " + sentence_id)

        sentence = sentence_data[sentence_id]

        # list of object labels that were hidden in the test data
        thisgold = golddata["imagine_scen"][sentence_id]

        # determine MAP scenarios for the sentence
        map_scenarios = thismodel["MAP"]["scenario"]

        yield( sentence_id, sentence, thisgold, map_scenarios)


##############3
# main 

random.seed(6543)

print("Evaluating")

# sentence ID -> model average precision, rank of highest correct prediction, perplexity
sentid_averageprecision = { }
sentid_highestcorrect = { }
sentid_perplexity = { }

# which sentences to sample for inspection?
sentence_ids_to_inspect = random.sample(list(sentid_sent.keys()), args.num_inspect)

# loop over files, evaluate, write inspection files
outpath = get_output_path(os.path.join(args.outdir, "eval_imagine_scen_out.txt"))
with open(outpath, "w") as outf:

    out_obj = sentence_util.SentencePrinter(vgindex_obj, file = outf)
    
    for sentence_id, sentence, gold_hidden_objectids, scenarios_this_sent in each_sentence(model_data, golddata, sentid_sent):

        model_objectid_ranking, model_objectid_logprob = scen_obj.predict_objectids(scenarios_this_sent)

        # print("gold hidden objects", [vgindex_obj.ix2l(o)[0] for o in gold_hidden_objectids])
        # print("scenarios", sorted(scenarios_this_sent))
        # for sc, _ in Counter(scenarios_this_sent).most_common(3):
        #    print(topic_obj.topic_info(sc))

        # print("model assign", [vgindex_obj.ix2l(o)[0] for o in model_objectid_ranking[:5]])
        
        
        sentid_averageprecision[ sentence_id ] = ( averageprecision(model_objectid_ranking, gold_hidden_objectids),
                                                   averageprecision(baseline_objectid_ranking, gold_hidden_objectids) )

        sentid_highestcorrect[ sentence_id] = (highestcorrect(model_objectid_ranking, gold_hidden_objectids),
                                               highestcorrect(baseline_objectid_ranking, gold_hidden_objectids) )

        sentid_perplexity[ sentence_id ] = ( perplexity(model_objectid_ranking, model_objectid_logprob, gold_hidden_objectids),
                                             perplexity(baseline_objectid_ranking, baseline_objectid_logprob, gold_hidden_objectids) )
                

        if sentence_id in sentence_ids_to_inspect:
            # write sentence to inspection file
            print("Sentence", sentence_id, file = outf)
            out_obj.write_sentence(sentence)
            
            print(file = outf)


            print("Omitted objects:", file = outf)
            print(", ".join([vgindex_obj.ix2l(oid)[0] for oid in gold_hidden_objectids]), file = outf)
            print(file = outf)

            print("Top predicted objects:", file = outf)
            print(", ".join([ vgindex_obj.ix2l(oid)[0] for oid in model_objectid_ranking[:15]]), file = outf)
            print(file = outf)

            print("Main scenarios:", file = outf)
            for sc, _ in Counter(scenarios_this_sent).most_common(3):
                print(sc, topic_obj.topic_info(sc), file = outf)
            
            print("\n\nAverage precision:", sentid_averageprecision[sentence_id][0], file = outf)
            print("Perplexity:", sentid_perplexity[sentence_id][0], file = outf)
            print(file = outf)

# compute mean average precision and report it
print("Mean average precision:", round(sum([m for m, b in sentid_averageprecision.values()]) / len(sentid_averageprecision), 3),
      "Baseline mean average precision:", round(sum([b for m, b in sentid_averageprecision.values()]) / len(sentid_averageprecision), 3))

# compute mean perplexity and report it
print("Average perplexity:", round(sum([m for m, b in sentid_perplexity.values()]) / len(sentid_perplexity), 3),
      "Baseline average perplexity:", round(sum([b for m, b in sentid_perplexity.values()]) / len(sentid_perplexity), 3))

print("Average rank of highest correct:", sum([m for m, b in sentid_highestcorrect.values()]) / len(sentid_highestcorrect),
      "Baseline average rank of highest correct", sum([b for m, b in sentid_highestcorrect.values()]) / len(sentid_highestcorrect))

print()

            
                
