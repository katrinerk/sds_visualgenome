# Katrin Erk January 2023

import os
import shutil
import sys
from pathlib import Path
from typing import List, Union
import configparser

#########
# includes path access functions
# by Pengxiang Cheng


class VGPaths:
    # store general paths to use
    def __init__(self, 
                vgdata = "data/",
                sdsdata = "sds_in/vanilla/",
                sdsout = "sds_out/vanilla/"):

        # read config file
        config = configparser.ConfigParser()
        config.read("settings.txt")
        
        self.visualgenome_path = config["VisualGenome"]["VGPATH"]
        self.vg_subpath_obj = config["VisualGenome"]["VGOBJECTS"]
        self.vg_subpath_attr = config["VisualGenome"]["VGATTRIB"]
        self.vg_subpath_rel = config["VisualGenome"]["VGREL"]
        
        self.datapath = vgdata
        self.sdsdatapath = sdsdata
        self.sdsoutpath = sdsout
        self.vg_manipulated_path = "changedvg/"


    # location of visual genome objects file
    def vg_objects_zip_and_filename(self, write = False):
        pathname = os.path.join(self.visualgenome_path, self.vg_subpath_obj)
        if write:
            return ( get_output_path(pathname), "objects.json")
        else:
            return (get_input_path(pathname), "objects.json")

    # location of visual genome attributes file
    def vg_attributes_zip_and_filename(self, write = False):
        pathname = os.path.join(self.visualgenome_path, self.vg_subpath_attr)
        if write:
            return ( get_output_path(pathname), "attributes.json")
        else:
            return (get_input_path(pathname), "attributes.json")

    # location of visual genome relations file
    def vg_relations_zip_and_filename(self, write = False):
        pathname = os.path.join(self.visualgenome_path, self.vg_subpath_rel)
        if write:
            return ( get_output_path(pathname), "relationships.json")
        else:
            return (get_input_path(pathname), "relationships.json")
    

    # where to store identities of sufficiently frequent objects, attributes, relations
    def vg_counts_zip_and_filename(self, write = False):
        pathname = os.path.join(self.datapath, "vgcounts.json.zip")
        if write:
            return( get_output_path(pathname), "vgcounts.json")
        else:
            return( get_input_path(pathname), "vgcounts.json")

    # where to store IDs of images in the training and test part
    def vg_traintest_zip_and_filename(self, write = False):
        pathname = os.path.join(self.datapath, "traintest.json.zip")
        if write:
            return( get_output_path(pathname), "traintest.json")
        else:
            return( get_input_path(pathname), "traintest.json")
        

    # where to find VG-based vectors
    def vg_vecfilename(self, write = False):
        pathname = os.path.join(self.datapath, "ext2vec.dm")
        if write:
            return get_output_path(pathname)
        else:
            return get_input_path(pathname)

    #######3
    # gensim related paths
    # where to store the gensim dictionary
    def gensim_dict_zip_and_filename(self, write = False):
        pathname = os.path.join(self.datapath, "gensim_dict.pkl.zip")

        if write:
            return( get_output_path(pathname), "gensim_dict.pkl")
        else:
            return( get_input_path(pathname), "gensim_dict.pkl")

    # where to store the gensim corpus
    def gensim_corpus_zipfilename(self, write = False):
        pathname = os.path.join(self.datapath, "gensim_corpus.zip")
        if write:
            return get_output_path(pathname)
        else:
            return get_input_path(pathname)

    # names of individual files in the corpus for gensim
    def gensim_corpus_filename(self, prefix):
        return str(prefix) + ".json"

    # gensim output
    def gensim_out_zip_and_filenames(self, write = False):
        pathname = os.path.join(self.datapath, "gensim_output.zip")
        if write:
            return (get_output_path(pathname), "overall.json", "topics.json", "words.json", "topicword.json")
        else:
            return ( get_input_path(pathname), "overall.json", "topics.json", "words.json", "topicword.json")

    ###########
    # files that will form input to SDS and output from SDS

    # zip files and inner files for parameter files for SDS
    def sds_filenames(self, write = False):

        f = get_output_path if write else get_input_path
        return {"general" : f(os.path.join(self.sdsdatapath, "general.json")),
                "selpref" : (f(os.path.join(self.sdsdatapath, "selpref.json.gzip")), "selpref.json"),
                "scenario_concept" : (f(os.path.join(self.sdsdatapath, "scenario_concept.json.gzip")), "scenario_concept.json"),
                "word_concept" : (f(os.path.join(self.sdsdatapath, "word_concept.json.gzip")), "word_concept.json")
                }

    # zip file for sentences as input to SDS
    def sds_sentence_zipfilename(self, write = False):
        pathname = os.path.join(self.sdsdatapath, "sentences.zip")
        if write:
            return (get_output_path(pathname), "results.json")
        else:
            return (get_input_path(pathname), "results.json")


    # zip file for where to write SDS output
    def sds_output_zipfilename(self, write = False):
        pathname = os.path.join(self.sdsoutpath, "sdsout.zip")
        if write:
            return (get_output_path(pathname), "results.json")
        else:
            return (get_input_path(pathname), "results.json")


    # zip file and inner file for gold information for tasks
    def sds_gold(self, write = False):
        pathname = os.path.join(self.sdsdatapath, "gold.json.gzip")
        if write:
            return (get_output_path(pathname), "gold.json")
        else:
            return (get_input_path(pathname), "gold.json")
    
    ########
    # where to put manipulated object, attribute, relation files
    # as input to vector creation
    def vg_manipulated_filenames(self, vg_manipulated_path, write = False):
        vgobjectfilename = os.path.basename(self.vg_subpath_obj)
        vgattribfilename = os.path.basename(self.vg_subpath_attr)
        vgrelfilename = os.path.basename(self.vg_subpath_rel)

        f = get_output_path if write else get_input_path
        
        return (f(os.path.join(vg_manipulated_path, vgobjectfilename)),
                f(os.path.join(vg_manipulated_path, vgattribfilename)),
                f(os.path.join(vg_manipulated_path, vgrelfilename)))





def get_input_path(path: Union[str, Path], check_exist: bool = True) -> Path:
    if isinstance(path, str):
        path = Path(path)
    if check_exist:
        assert path.exists(), f'{path} does not exist!'

    return path


def get_output_path(path: Union[str, Path], overwrite_warning: bool = True) -> Path:
    if isinstance(path, str):
        path = Path(path)
    if path.exists() and overwrite_warning:
        if not query_yes_no(f'{path} already exists, overwrite?', default='yes'):
            sys.exit(0)

    path.parent.mkdir(parents=True, exist_ok=True)

    return path


def get_output_dir(dir_path: Union[str, Path], overwrite_warning: bool = True) -> Path:
    if isinstance(dir_path, str):
        dir_path = Path(dir_path)

    if dir_path.is_dir() and any(True for _ in dir_path.iterdir()):
        if overwrite_warning:
            if not query_yes_no(f'{dir_path} already exists and is not empty, delete it first?',
                                default='yes'):
                sys.exit(0)
        shutil.rmtree(str(dir_path.resolve()))

    dir_path.mkdir(exist_ok=True, parents=True)

    return dir_path


def get_file_list(path: Union[str, Path], suffix: str = None, sort: bool = True,
                  ignore_hidden: bool = True) -> List[Path]:
    if isinstance(path, str):
        path = get_input_path(path)
    if path.is_file():
        file_list = [path]
    else:
        if suffix is None:
            file_list = [f for f in path.iterdir()]
        else:
            file_list = [f for f in path.glob('*{}'.format(suffix))]
        if sort:
            file_list = sorted(file_list)

    if ignore_hidden:
        file_list = [f for f in file_list if not f.name.startswith('.')]

    return file_list


def query_yes_no(question, default=None):
    """
    Ask a yes/no question via input() and return their answer.

    :param question: a string that is presented to the user
    :param default: the presumed answer if the user just hits <Enter>.
    It must be 'yes', 'no' or None (the default, meaning an answer is required of the user).
    :return: True or False
    """
    valid = {'yes': True, 'y': True, 'ye': True, 'no': False, 'n': False}
    if default is None:
        prompt = ' [y/n] '
    elif default == 'yes':
        prompt = ' [Y/n] '
    elif default == 'no':
        prompt = ' [y/N] '
    else:
        raise ValueError(f'invalid default answer: {default}')

    while 1:
        print(question + prompt, end='')
        choice = input().lower()
        if default is not None and choice == '':
            return default
        elif choice in valid.keys():
            return valid[choice]
        else:
            print('Please respond with \'yes\' or \'no\' (or \'y\' or \'n\')')


