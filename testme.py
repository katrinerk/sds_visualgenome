

import sys
import os
import json
import zipfile
from collections import defaultdict, Counter
import numpy as np

from vgnames import VGOBJECTS, VGATTRIBUTES, VGRELATIONS 
import vgiterator
import sentence_util
from vgindex import VgitemIndex
from vgpaths import VGPaths, get_output_path
from sentence_util import SentencePrinter


########


vgpath_obj = VGPaths(sdsdata = "sds_in/discourse")

# read frequent object counts
vgcounts_zipfilename, vgcounts_filename = vgpath_obj.vg_counts_zip_and_filename()
with zipfile.ZipFile(vgcounts_zipfilename) as azip:
    with azip.open(vgcounts_filename) as f:
        vgobjects_attr_rel = json.load(f)

# read  gold data
zipfilename, filename = vgpath_obj.sds_gold()
with zipfile.ZipFile(zipfilename) as azip:
    with azip.open(filename) as f:
        golddata = json.load(f)

    
# read passages
zipfilename, filename = vgpath_obj.sds_sentence_zipfilename()
with zipfile.ZipFile(zipfilename) as azip:
    with azip.open(filename) as f:
        passages = json.load(f)

vgindex_obj = VgitemIndex(vgobjects_attr_rel)        
out_obj = SentencePrinter(vgindex_obj)

for passage in passages:
    print("************************")
    for sent_obj in passage:
        print("--Sentence--", sent_obj["sentence_id"])
        out_obj.write_sentence(sent_obj["sentence"])

