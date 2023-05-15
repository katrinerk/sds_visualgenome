import concurrent.futures
import sys
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

from sds_core import SDS, onesentence_map
from vgpaths import VGPaths

if __name__ == '__main__':
    # command line arguments
    parser = ArgumentParser()
    parser.add_argument('input', help="directory with input sentences")
    parser.add_argument('output', help="directory for system output")    

    args = parser.parse_args()

    # settings file
    config = configparser.ConfigParser()
    config.read("settings.txt")

    vgpath_obj = VGPaths(sdsdata = args.input, sdsout = args.output)

    # make SDS object
    sds_obj = SDS(vgpath_obj, config["Scenarios"])

    # store MAP results and marginals
    results = [ ]
    counter = 0

    # process each utterance go through utterances in the input file,
    sentlength_runtime = defaultdict(list)


    for sentence_id, sentence in sds_obj.each_sentence_json(verbose = False):

        sentlength = sum(1 for ell in sentence if ell[0] == "w")

        # make a timer to see how long it takes to do
        # graph creation + inference
        tic=timeit.default_timer()
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            thismap = executor.submit(onesentence_map, sentence_id, sentence, sds_obj).result()


        if thismap is not None:
            results.append({"sentence_id" : sentence_id, "MAP" : thismap})

        # stop timer
        toc=timeit.default_timer()
        sentlength_runtime[sentlength].append(toc - tic)
        
        counter += 1
        if counter % 20 == 0: print(counter)


    # write results
    # into the output zip file
    zipfilename, filename = vgpath_obj.sds_output_zipfilename( write = True)
    with zipfile.ZipFile(zipfilename, "w", zipfile.ZIP_DEFLATED) as azip:
        azip.writestr(filename, json.dumps(results))                

    # write average runtimes by sentence length
    for ell in sorted(sentlength_runtime.keys()):
        print("Sentence length", ell, "#sentences", len(sentlength_runtime[ell]),
              "mean runtime", round(statistics.mean(sentlength_runtime[ell]), 3))
