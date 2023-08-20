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

from sdsd_core import SDSD, onediscourse_map
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
    sds_obj = SDSD(vgpath_obj, config["Scenarios"])

    # store MAP results and marginals
    results = [ ]
    counter = 0

    # process each utterance go through utterances in the input file,
    passagelength_runtime = [ ]

    for passage in  sds_obj.each_passage_json(verbose = False):
        numwords = sum([len(sent) for sentid, sent in passage])
        
        # make a timer to see how long it takes to do
        # graph creation + inference
        tic=timeit.default_timer()

        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            result = executor.submit(onediscourse_map, passage, sds_obj).result()


        # stop timer
        toc=timeit.default_timer()
        passagelength_runtime.append( (numwords / len(passage), toc - tic) )
        
        if result is not None:
            thismap, thismentalfile, thiscoref = result
            results.append({"sentids" : [sentid for sentid, _ in passage], "MAP" : thismap, "mentalfiles" : thismentalfile, "coref" : thiscoref})


        # print("HIER len passage", len(passage), [sentid for sentid, _ in passage])
        # sys.exit(0)
        counter += 1
        if counter % 10 == 0: print(counter)


    # write results
    # into the output zip file
    zipfilename, filename = vgpath_obj.sds_output_zipfilename( write = True)
    with zipfile.ZipFile(zipfilename, "w", zipfile.ZIP_DEFLATED) as azip:
        azip.writestr(filename, json.dumps(results))                

    # show average runtimes by sentence length
    if len(passagelength_runtime) > 0:
        print("Average sentence length", round(statistics.mean([length for length, _ in passagelength_runtime]), 3))
        print("Mean runtime", round(statistics.mean([runtime for _, runtime in passagelength_runtime]), 3))
