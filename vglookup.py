# Katrin Erk May 2023
# look up IDs for VG words


import sys
import os
import json
import zipfile

import vgiterator
from sds_input_util import VGSentences, VGParam
from vgindex import VgitemIndex
from vgpaths import VGPaths

vgpath_obj = VGPaths()

vgcounts_zipfilename, vgcounts_filename = vgpath_obj.vg_counts_zip_and_filename()
with zipfile.ZipFile(vgcounts_zipfilename) as azip:
    with azip.open(vgcounts_filename) as f:
        vgobjects_attr_rel = json.load(f)
        
vgindex_obj = VgitemIndex(vgobjects_attr_rel)

while True:
    word = input("Word to look up (or q to quit): ")
    if word == "q":
        break
    wtype = input("Word type: o/a/r? ")

    if wtype == "o":
        print(vgindex_obj.o2ix(word))
    elif wtype == "a":
        print(vgindex_obj.a2ix(word))
    elif wtype == "r":
        print(vgindex_obj.r2ix(word))

    
