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

########3
# settings


parser = ArgumentParser()
parser.add_argument('--sdsinput', help="directory with input to sds, default: sds_in/imagine_scen", default = "sds_in/imagine_scen/")
parser.add_argument('--sdsoutput', help="directory with sds output, default: sds_out/imagine_scen", default = "sds_out/imagine_scen/")
parser.add_argument('--outdir', help="directory to write output for inspection, default: inspect_output/imagine_scen", default = "inspect_output/imagine_scen/")
parser.add_argument('--vgdata', help="directory with VG data including frequent items, train/test split, topic model", default = "data/")
parser.add_argument('--numsent_inspect', help = "number of sentences to write for inspection, default 10", type = int, default = 10)

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


##########################
# for each scenario, obtain a numpy array
# of object log probabilities. to do this,
# kick out attribute and relation log probabilities and renormalize

print("reading scenario/concept probabilities")

# read scenario/topic data from gensim.
# format: one row per topic, one column per word, log probabilities
gensim_zipfilename, overall_filename, topic_filename, word_filename, topicword_filename = vgpath_obj.gensim_out_zip_and_filenames()
with zipfile.ZipFile(gensim_zipfilename) as azip:
    # term-topic matrix
    with azip.open(topicword_filename) as f:
        term_topic_matrix = json.load(f)

    # ordered list of words as they appear in topics,
    # need to be mapped to IDs
    # ordered list of words as they appear in topics. need to be mapped to concepts and concept indices
    with azip.open(word_filename) as f:
        topic_wordlist = json.load(f)

# make list of indices of gensim words that are objects,
# and map object names ot their IDs
indices_of_objects = [ ]
object_ids = [ ]
for index, word in enumerate(topic_wordlist):
    if word[:3] =="obj":
        wordid = vgindex_obj.o2ix(word[3:])
        if wordid is None:
            raise Exception("lookup error for topic word, object", word[3:])

        object_ids.append(wordid)
        indices_of_objects.append(index)

indices_of_objects = np.array(indices_of_objects)

# for each scenario, restrict to objects, renormalize, store
scenario_logprobs = [ ]
for sclp in term_topic_matrix:
    
    a = np.array(sclp)
    
    # select only columns that describe an object
    a = a[ indices_of_objects ]

    # renormalize
    normalizer = np.log(np.exp(a).sum())
    a = a - normalizer
    # scenario_logprobs: scenario log probabilities, in order of scenarios
    scenario_logprobs.append(a)

topic_obj = sentence_util.TopicInfoUtil(vgpath_obj)

# for sc, logprobs in enumerate(scenario_logprobs):
    
#     print("----")
#     print(topic_obj.topic_info(sc))
#     print([vgindex_obj.ix2l(o)[0] for o in np.array(object_ids)[logprobs.argsort()][::-1][:10]])
    
#################################
# evaluation: first do the actual prediction of additional objects based on the MAP scenario assignment,
# then compute mean average precision

##############
# for a given list of scenarios,
# determine probability of sampling each possible object ID
def predict_objectids(scenario_list, scenario_logprobs, object_ids):
    # obtain scenario probabilities within the scenario list
    sc_freq = Counter(scenario_list)
    sc_logprob = dict( (s, math.log( sc_freq[s] / sc_freq.total())) for s in sc_freq.keys() )

    # sc_logprob: dictionary scenario ID -> log probability of this scenario in this sentence
    # scenario_logprobs: list where the i-th entry is for the i-th scenario,
    #    each entry is a numpy array of object log probabilities
    # compute: sum_{s scenario in this sentence} this_sentence_scenarioprob(s) * object_probs(s)
    # again a numpy array of object log probabilities
    objectid_logprob = np.sum([ sc_logprob[s] + scenario_logprobs[s] for s in sc_logprob.keys()], axis = 0)

    # sort object IDs by their log probability in objectid_logprob,
    # largest first
    return np.array(object_ids)[objectid_logprob.argsort()][::-1]

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
# main loop

random.seed(6543)

print("Evaluating")

# sentence ID -> (model average precision, predicted average precision)
sentid_averageprecision = { }
sentid_highestcorrect = { }

# which sentences to sample for inspection?
sentence_ids_to_inspect = random.sample(list(sentid_sent.keys()), args.numsent_inspect)

# loop over files, evaluate, write inspection files
outpath = get_output_path(os.path.join(args.outdir, "eval_imagine_scen_out.txt"))
with open(outpath, "w") as outf:

    out_obj = sentence_util.SentencePrinter(vgindex_obj, file = outf)
    
    for sentence_id, sentence, gold_hidden_objectids, scenarios_this_sent in each_sentence(model_data, golddata, sentid_sent):

        model_objectid_ranking = predict_objectids(scenarios_this_sent, scenario_logprobs, object_ids)

        # print("gold hidden objects", [vgindex_obj.ix2l(o)[0] for o in gold_hidden_objectids])
        # print("scenarios", sorted(scenarios_this_sent))
        # for sc, _ in Counter(scenarios_this_sent).most_common(3):
        #    print(topic_obj.topic_info(sc))

        # print("model assign", [vgindex_obj.ix2l(o)[0] for o in model_objectid_ranking[:5]])
        
        
        sentid_averageprecision[ sentence_id ] = ( averageprecision(model_objectid_ranking, gold_hidden_objectids),
                                                   averageprecision(baseline_objectid_ranking, gold_hidden_objectids) )

        sentid_highestcorrect[ sentence_id] = (highestcorrect(model_objectid_ranking, gold_hidden_objectids),
                                               highestcorrect(baseline_objectid_ranking, gold_hidden_objectids) )
                

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
            
            print("Average precision:", sentid_averageprecision[sentence_id][0], file = outf)
            print(file = outf)

# compute mean average precision and report it
print("Mean average precision:", round(sum([m for m, b in sentid_averageprecision.values()]) / len(sentid_averageprecision), 3),
      "Baseline mean average precision:", round(sum([b for m, b in sentid_averageprecision.values()]) / len(sentid_averageprecision), 3))


# print("Average rank of highest correct:", sum([m for m, b in sentid_highestcorrect.values()]) / len(sentid_highestcorrect),
#      "Baseline average rank of highest correct", sum([b for m, b in sentid_highestcorrect.values()]) / len(sentid_highestcorrect))
            
                
